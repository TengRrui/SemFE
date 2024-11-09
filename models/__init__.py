from models.model import MatchingNet

def build_model(args):
    return MatchingNet(
        args.d_coarse_model,
        args.d_fine_model,
        args.matching_name,
        args.match_threshold,
        args.window_size,
        args.border,
        args.sinkhorn_iterations
    )
