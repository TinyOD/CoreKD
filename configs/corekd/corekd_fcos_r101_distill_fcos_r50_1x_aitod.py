_base_ = [
    '../../fcos/fcos_center-normbbox-giou_r50_caffe_fpn_gn-head_mstrain_1x_aitod.py',
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_corekd=1.25e-3
beta_corekd=0.5e-3
epsilon_corekd=0.6e-6

distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_corekd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_corekd=alpha_corekd,
                                       beta_corekd=beta_corekd,
                                       epsilon_corekd=epsilon_corekd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_corekd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_corekd=alpha_corekd,
                                       beta_corekd=beta_corekd,
                                       epsilon_corekd=epsilon_corekd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_corekd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_corekd=alpha_corekd,
                                       beta_corekd=beta_corekd,
                                       epsilon_corekd=epsilon_corekd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_corekd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_corekd=alpha_corekd,
                                       beta_corekd=beta_corekd,
                                       epsilon_corekd=epsilon_corekd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_corekd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_corekd=alpha_corekd,
                                       beta_corekd=beta_corekd,
                                       epsilon_corekd=epsilon_corekd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/fcos/fcos_center-normbbox-giou_r50_caffe_fpn_gn-head_mstrain_1x_coco.py'
teacher_cfg = 'configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(step=[8, 11])
