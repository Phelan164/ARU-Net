from __future__ import print_function, division

import click
import os

from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.util.validator_pb import Validator_pb


@click.command()
@click.option('--path_list_val', default="./data/test.lst")
@click.option('--restore_pb_path', default="./models/linecut.pb")
def run(path_list_val, restore_pb_path):
    # kwargs_dat=dict(scale_val=0.33, one_hot_encoding=True, shuffle=False)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    # Images have to be gray scale images
    img_channels = 1
    # Number of output classes
    n_class = 2
    kwargs_dat = dict(scale_val=0.33, one_hot_encoding=True)
    data_provider = Data_provider_la(None, path_list_val, n_class, threadNum=1, kwargs_dat=kwargs_dat)
    cost_kwargs = dict(cost_name="cross_entropy")
    validator = Validator_pb(restore_pb_path, n_class=n_class, cost_kwargs=cost_kwargs)
    validator.validate(data_provider)


if __name__ == '__main__':
    run()