import sys
from condot.train.experiment import prepare
from ccot.evaluate.evaluate import eval_fxn
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("config", "", "Path to config")
flags.DEFINE_string("exp_group", "condot_exps", "Name of experiment.")
flags.DEFINE_boolean("restart", False, "delete cache")
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_boolean("dry", False, "dry mode")
flags.DEFINE_boolean("verbose", False, "run in verbose mode")

def main(argv):
    config, outdir = prepare(argv)

    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    try:
        eval_fxn(outdir, config)
    except ValueError as error:
        print("Training bugged")
        raise error
    else:
        print("Training finished")

    return


if __name__ == "__main__":
    main(sys.argv)