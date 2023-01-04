from .cam_data import gen_calculate
from .cam_data import save as save_cam_data
from .cam_data import load as load_cam_data


from .output_masked import save_ouput_model_data
from .output_masked import generator_output_model


def load_model_from_run(run, model_name= 'model-best.h5'):
    path = run.file(model_name).download(replace=True).name
    model = load_model(path)
    return model


def get_path_run(run,root_path):
    run_id = '_'.join([*run.path,run.config['model']])
    folder_path = os.path.join(root_path, run_id)
    return folder_path