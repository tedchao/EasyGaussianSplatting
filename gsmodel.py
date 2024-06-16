import torch
import gsplatcu as gsc
from gsplat.utils import *


class GSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pws,
        shs,
        alphas,
        scales,
        rots,
        us,
        cam,
    ):
        # more detail view forward.pdf
        # step1. Transform pw to camera frame,
        # and project it to iamge.
        us, pcs, depths, du_dpcs = gsc.project(
            pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # step2. Calcuate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = gsc.computeCov3D(
            rots, scales, depths, True)

        # step3. Calcuate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = gsc.computeCov2D(
            cov3ds, pcs, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # step4. get color info
        colors, dcolor_dshs, dcolor_dpws = gsc.sh2Color(shs, pws, cam.twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, depths, True)
        image, contrib, final_tau, patch_range_per_tile, gsid_per_patch =\
            gsc.splat(cam.height, cam.width,
                      us, cinv2ds, alphas, depths, colors, areas)

        # Store the static parameters in the context
        ctx.cam = cam
        # Keep relevant tensors for backward
        ctx.save_for_backward(us, cinv2ds, alphas,
                              depths, colors, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dcov3ds,
                              dcov3d_drots, dcov3d_dscales, dcolor_dshs,
                              du_dpcs, dcov2d_dpcs, dcolor_dpws)
        return image, areas

    @staticmethod
    def backward(ctx, dloss_dgammas, _):
        # Retrieve the saved tensors and static parameters
        cam = ctx.cam
        us, cinv2ds, alphas, \
            depths, colors, contrib, final_tau,\
            patch_range_per_tile, gsid_per_patch,\
            dcinv2d_dcov2ds, dcov2d_dcov3ds,\
            dcov3d_drots, dcov3d_dscales, dcolor_dshs,\
            du_dpcs, dcov2d_dpcs, dcolor_dpws = ctx.saved_tensors

        # more detail view backward.pdf

        # section.5
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        dpc_dpws = cam.Rcw
        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds
        # backward.pdf equation (3)
        dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        # backward.pdf equation (4)
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        # backward.pdf equation (5)
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @
                      dcolor_dshs).permute(0, 2, 1).squeeze()

        dloss_dshs = dloss_dshs.reshape(dloss_dshs.shape[0], -1)
        # backward.pdf equation (7)
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws

        return dloss_dpws.squeeze(),\
            dloss_dshs,\
            dloss_dalphas.squeeze().unsqueeze(1),\
            dloss_dscales.squeeze(),\
            dloss_drots.squeeze(),\
            dloss_dus.squeeze(),\
            None


def get_gs_params(gs):
    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = get_scales_raw(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = get_alphas_raw(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda')
    low_shs = shs[:, :3]
    high_shs = torch.ones_like(low_shs).repeat(1, 15) * 0.001
    high_shs[:, :shs[:, 3:].shape[1]] = shs[:, 3:]
    low_shs = low_shs.requires_grad_()
    high_shs = high_shs.requires_grad_()
    gs_params = {"pws": pws, "low_shs": low_shs, "high_shs": high_shs,
                 "alphas_raw": alphas_raw, "scales_raw": scales_raw, "rots_raw": rots_raw}
    return gs_params


def update_params(optimizer, gs_params, gs_params_new):
    for group in optimizer.param_groups:
        param_new = gs_params_new[group["name"]]
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = torch.cat(
                (state["exp_avg"], torch.zeros_like(param_new)), dim=0)
            state["exp_avg_sq"] = torch.cat(
                (state["exp_avg_sq"], torch.zeros_like(param_new)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], param_new), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = state
            gs_params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], param_new), dim=0).requires_grad_(True))
            gs_params[group["name"]] = group["params"][0]


def prune_params(optimizer, gs_params, mask):
    for group in optimizer.param_groups:
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = state["exp_avg"][mask]
            state["exp_avg_sq"] = state["exp_avg_sq"][mask]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                (group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = state
            gs_params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                group["params"][0][mask].requires_grad_(True))
            gs_params[group["name"]] = group["params"][0]


class GSModel(torch.nn.Module):
    def __init__(self, sense_size, max_steps):
        super().__init__()
        self.cunt = None
        self.grad_accum = None
        self.cam = None
        self.grad_threshold = 0.0002 / 50
        self.scale_threshold = 0.01 * sense_size
        self.alpha_threshold = 0.005
        self.big_threshold = 0.1 * sense_size
        self.reset_alpha_val = 0.01
        self.iteration = 0
        self.pws_lr_scheduler = get_expon_lr_func(lr_init=1e-4 * sense_size,
                                                lr_final=1e-6 * sense_size,
                                                lr_delay_mult=0.01,
                                                max_steps=max_steps)

    def forward(
            self,
            pws,
            low_shs,
            high_shs,
            alphas_raw,
            scales_raw,
            rots_raw,
            us,
            cam):
        self.cam = cam
        # Limit the value of alphas: 0 < alphas < 1
        alphas = get_alphas(alphas_raw)
        # Limit the value of scales > 0
        scales = get_scales(scales_raw)
        # Limit the value of rot, normal of rots is 1
        rots = get_rots(rots_raw)

        shs = get_shs(low_shs, high_shs)

        # apply GSfunction (forward)
        image, areas = GSFunction.apply(pws, shs, alphas, scales, rots, us, cam)

        return image, areas

    def update_density_info(self, dloss_dus, areas):
        with torch.no_grad():
            visible = areas[:, 0] > 0
            dloss_dus[:, 0] = dloss_dus[:, 0]
            dloss_dus[:, 0] = dloss_dus[:, 1]
            grad = torch.norm(dloss_dus, dim=-1, keepdim=True)

            if self.cunt is None:
                self.grad_accum = grad
                self.cunt = visible.to(torch.int32)
            else:
                self.cunt += visible
                self.grad_accum[visible] += grad[visible]
            pass

    def update_gaussian_density(self, gs_params, optimizer):
        # prune too small or too big gaussian
        selected_by_small_alpha = gs_params["alphas_raw"].squeeze() < get_alphas_raw(self.alpha_threshold)
        selected_by_big_scale = torch.max(gs_params["scales_raw"], axis=1)[0] > get_scales_raw(self.big_threshold)
        selected_for_prune = torch.logical_or(selected_by_small_alpha, selected_by_big_scale)
        selected_for_remain = torch.logical_not(selected_for_prune)
        prune_params(optimizer, gs_params, selected_for_remain)

        grads = self.grad_accum.squeeze()[selected_for_remain] / self.cunt[selected_for_remain]
        grads[grads.isnan()] = 0.0

        pws = gs_params["pws"]
        low_shs = gs_params["low_shs"]
        high_shs = gs_params["high_shs"]
        alphas = get_alphas(gs_params["alphas_raw"])
        scales = get_scales(gs_params["scales_raw"])
        rots = get_rots(gs_params["rots_raw"])

        selected_by_grad = grads >= self.grad_threshold
        selected_by_scale = torch.max(scales, axis=1)[0] <= self.scale_threshold

        selected_for_clone = torch.logical_and(selected_by_grad, selected_by_scale)
        selected_for_split = torch.logical_and(selected_by_grad, torch.logical_not(selected_by_scale))

        # clone gaussians
        pws_cloned = pws[selected_for_clone]
        low_shs_cloned = low_shs[selected_for_clone]
        high_shs_cloned = high_shs[selected_for_clone]
        alphas_cloned = alphas[selected_for_clone]
        scales_cloned = scales[selected_for_clone]
        rots_cloned = rots[selected_for_clone]

        # split gaussians
        # try:
        #     Cov3d = compute_cov_3d_torch(
        #         scales[selected_for_split], rots[selected_for_split])
        #     multi_normal = torch.distributions.MultivariateNormal(
        #         loc=pws[selected_for_split], covariance_matrix=Cov3d)
        #     pws_splited = multi_normal.sample()  # sampling new pw for splited gaussian
        # except Exception as e:
        #     print(e)

        rots_splited = rots[selected_for_split]
        means = torch.zeros((rots_splited.size(0), 3), device="cuda")
        stds = scales[selected_for_split]
        samples = torch.normal(mean=means, std=stds)
        # sampling new pw for splited gaussian
        pws_splited = pws[selected_for_split] + \
            rotate_vector_by_quaternion(rots_splited, samples)
        alphas_splited = alphas[selected_for_split]
        scales[selected_for_split] = scales[selected_for_split] * 0.6  # splited gaussian will go smaller
        scales_splited = scales[selected_for_split]
        low_shs_splited = low_shs[selected_for_split]
        high_shs_splited = high_shs[selected_for_split]

        gs_params_new = {"pws": torch.cat([pws_cloned, pws_splited]),
                         "low_shs": torch.cat([low_shs_cloned, low_shs_splited]),
                         "high_shs": torch.cat([high_shs_cloned, high_shs_splited]),
                         "alphas_raw": get_alphas_raw(torch.cat([alphas_cloned, alphas_splited])),
                         "scales_raw": get_scales_raw(torch.cat([scales_cloned, scales_splited])),
                         "rots_raw": torch.cat([rots_cloned, rots_splited])}

        # debug = True
        # if (debug):
        #     rgb = torch.Tensor([1, 0, 0]).to(torch.float32).to('cuda')
        #     flat_shs = (rgb - 0.5) / 0.28209479177387814
        #     flat_shs = flat_shs.repeat(gs_params_new['pws'].shape[0], 1)
        #     gs_params_new['shs'] = flat_shs
        #     debug_gs = {"pws": torch.cat([gs_params["pws"], gs_params_new["pws"]]),
        #                 "shs": torch.cat([gs_params["shs"], gs_params_new["shs"]]),
        #                 "alphas_raw": torch.cat([gs_params["alphas_raw"], gs_params_new["alphas_raw"]]),
        #                 "scales_raw": torch.cat([gs_params["scales_raw"], gs_params_new["scales_raw"]]),
        #                 "rots_raw": torch.cat([gs_params["rots_raw"], gs_params_new["rots_raw"]])}

        # split gaussians (N is the split size)
        update_params(optimizer, gs_params, gs_params_new)
        print("---------------------")
        print("gaussian density update report")
        prune_n = int(torch.sum(selected_for_prune))
        clone_n = int(torch.sum(selected_for_clone))
        split_n = int(torch.sum(selected_for_split))
        print("pruned num: ", prune_n)
        print("cloned num: ", clone_n)
        print("splited num: ", split_n)
        print("total gaussian number: ", gs_params['pws'].shape[0])
        print("---------------------")

        self.grad_accum = None
        self.cunt = None

        # if (debug):
        #     return debug_gs

    def reset_alpha(self, gs_params, optimizer):
        reset_alpha_raw_val = get_alphas_raw(self.reset_alpha_val)
        rest_mask = gs_params['alphas_raw'] > reset_alpha_raw_val
        gs_params['alphas_raw'][rest_mask] = torch.ones_like(gs_params['alphas_raw'])[rest_mask] * reset_alpha_raw_val

        alpha_param = list(filter(lambda x: x["name"] == "alphas_raw", optimizer.param_groups))[0]

        state = optimizer.state.get(alpha_param['params'][0], None)
        state["exp_avg"] = torch.zeros_like(gs_params['alphas_raw'])
        state["exp_avg_sq"] = torch.zeros_like(gs_params['alphas_raw'])
        del optimizer.state[alpha_param['params'][0]]
        optimizer.state[alpha_param['params'][0]] = state

    def update_pws_lr(self, optimizer):
        pws_lr = self.pws_lr_scheduler(self.iteration)
        pws_param = list(filter(lambda x: x["name"] == "pws", optimizer.param_groups))[0]
        pws_param['lr'] = pws_lr
        self.iteration += 1