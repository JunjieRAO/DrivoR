import torch.nn as nn
# from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
# from ..bevformer.transformer_decoder import MLP
from ..layers.utils.mlp import MLP
# from .map_head import MapHead

METRIC_ORDER = (
    'no_at_fault_collisions',
    'drivable_area_compliance',
    'time_to_collision_within_bound',
    'ego_progress',
    'driving_direction_compliance',
    'comfort',
)


class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = len(METRIC_ORDER)

        # self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)


        self.pred_score = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'drivable_area_compliance':
                nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, 1),
                ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(config.tf_d_model,config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1),
            ),
            # 'lane_keeping': nn.Sequential(
            #     nn.Linear(d_model, d_ffn),
            #     nn.ReLU(),
            #     nn.Linear(d_ffn, 1),
            # ),
            # 'traffic_light_compliance': nn.Sequential(
            #     nn.Linear(config.tf_d_model, config.tf_d_ffn),
            #     nn.ReLU(),
            #     nn.Linear(config.tf_d_ffn, 1),
            # ),
            'comfort': nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, 1)
            )
        })





        self.double_score=config.double_score

        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred= config.agent_pred

        if self.agent_pred:
            if self.b2d:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2*6* 9)
            else:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn,2* 40 * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if config.one_token_per_traj:
                if self.b2d:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, config.num_poses*2)
                else:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, config.num_poses*5*2)
            else:
                if self.b2d:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
                else:
                    self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)

        self.bev_map=config.bev_map
        self.bev_agent=config.bev_agent

        if config.bev_agent:
            raise NotImplementedError
            self._agent_head=MyTransformeDecoder(config,config.num_bounding_boxes,6,trajenc=False)

        if config.bev_map:
            raise NotImplementedError
            self.map_head=MapHead(config)

        self.poses_num=config.num_poses
        self.tf_d_model=config.tf_d_model
        self.one_token_per_traj = config.one_token_per_traj


    def forward(self, proposals, metric_features):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        expected_prefix = (batch_size, p_size, self.score_num)
        if metric_features.ndim != 4 or tuple(metric_features.shape[:3]) != expected_prefix:
            raise ValueError(
                f"Expected metric_features shape [{batch_size}, {p_size}, "
                f"{self.score_num}, D], "
                f"got {tuple(metric_features.shape)}"
            )

        proposal_feature = metric_features.mean(dim=2)
        pred_logit = {}
        
        # selected_indices: B,
        for metric_idx, metric_name in enumerate(METRIC_ORDER):
            pred_logit[metric_name] = self.pred_score[metric_name](
                metric_features[:, :, metric_idx, :]
            ).squeeze(-1)
        
        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, self.score_num)

        if  self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(proposal_feature)
                if self.one_token_per_traj:
                    pred_area_logit = pred_area_logit.reshape(batch_size,p_size,t_size,-1)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size,p_size,t_size,-1,2,9)

            if self.bev_map:
                bev_semantic_map = self.map_head(proposal_feature)

            if self.bev_agent:
                agents =self._agent_head(None,proposal_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return pred_logit, pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels
