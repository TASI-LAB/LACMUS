testing_adv_paths = [{"name": "fgsm", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/detectors/adv_data/mnist_fgsm_0.03125.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/mnist_labels/mnist_org.npy"},
                         {"name": "pgd", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/detectors/adv_data/mnist_pgd1_5.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/mnist_labels/mnist_org.npy"},
                         {"name": "sa", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/detectors/adv_data/mnist_sa.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/mnist_labels/mnist_org.npy"},
                         {"name": "sta", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/detectors/adv_data/mnist_sta.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/mnist_labels/mnist_org.npy"},
                         {"name": "mix", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/detectors_review/detectors/adv_data/mnist_mix.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/mnist_c/mix_labels_100.npy"},
                         {"name": "ours", "images_numpy": "/bowen/d61-ai-security/work/zho086/dnn-gp/LACMUS/aug_sample/aug_sample.npy", "labels": "/bowen/d61-ai-security/work/zho086/dnn-gp/LACMUS/aug_sample/aug_labels.npy"},
                         
                         ]

def get_adv_path():
    return testing_adv_paths