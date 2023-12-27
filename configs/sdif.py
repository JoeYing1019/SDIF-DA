class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        common_parameters = {
            'data_mode': 'multi-class',
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'acc',
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 10
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        hyper_parameters = {
            'num_train_epochs': 100,
            'dst_feature_dims': 768,
            'n_levels_self': 1,
            'n_levels_cross': 1,
            'dropout_rate': 0.2,
            'cross_dp_rate': 0.3,
            'cross_num_heads': 12,
            'self_num_heads': 8,
            'grad_clip': 7, 
            'lr': 9e-6,
            'opt_patience': 8,
            'factor': 0.5,
            'weight_decay': 0.01,
            'aug_lr': 1e-6,
            'aug_epoch': 1,
            'aug_dp': 0.3,
            'aug_weight_decay': 0.1,
            'aug_grad_clip': -1.0,
            'aug_batch_size': 16,
            'aug': True
        }
        return hyper_parameters