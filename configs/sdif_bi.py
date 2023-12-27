class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        common_parameters = {
            'data_mode': 'binary-class',
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
            'n_levels_self': 2,
            'n_levels_cross': 3,
            'dropout_rate': 0.1,
            'cross_dp_rate': 0.3,
            'cross_num_heads': 2,
            'self_num_heads': 1,
            'grad_clip': 1, 
            'lr': 8e-6,
            'opt_patience': 4,
            'factor': 0.2,
            'weight_decay': 0.03,
            'aug_lr': 9e-7,
            'aug_epoch': 5,
            'aug_dp': 0,
            'aug_weight_decay': 0.1,
            'aug_grad_clip': 1.0,
            'aug_batch_size': 32,
            'aug': True
        }
        return hyper_parameters