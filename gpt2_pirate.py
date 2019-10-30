import argparse
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path

import tensorflow as tf

from GPT2_models.inputs import *
from GPT2_models.model_fns import *
from GPT2_models.predict_fns import *

# This program was designed to function with multiple kinds of models, but currently only GPT2 is supported
# The first element in the tupel is the model function, the second is the function called when predicting
models = {
    "GPT2": (gpt2_model, gpt2_predict)
}

inputs = {
    "openwebtext": openwebtext, # Standard OpenWebtext input
    "openwebtext_longbiased": openwebtext_longbiased, # OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": openwebtext_long, # Openwebtext that only shows long examples
}


def Our_gpt2(model, input_file, input_text, topk):
    """     parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str) # JSON file that contains model parameters
    parser.add_argument("--predict_file", type=str) # File to take as input for predict
    parser.add_argument("--predict_text", type=str) # Take string directly from args
    parser.add_argument("--top_k", type=int) # Top K truncation parameter for text generation
    args = parser.parse_args() """
    # Get prediction text
    predict_mode = False
    if input_file is not None:
        predict_mode = True
        with open(input_file) as f:
            text = f.read()
    elif input_text is not None:
        predict_mode = True
        text = input_text
    elif input_file is not None and input_text is not None:
        print("ERROR: 你他媽能不能只放一個輸入???")
        sys.exit()


    # Setup logging
    Path("GPT2_models/logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('GPT2_models/logs/PrettyBig.json.log'), # hard-coded path
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers

    # Read params of model
    with open(model, "r") as f:
        params = json.load(f)

    """     if not args.tpu is None:
        params["use_tpu"] = True
    else:
        params["use_tpu"] = False """
    params["use_tpu"] = False

    if topk is not None:
        params["top_k"] = topk

    if not "precision" in params.keys():
        params["precision"] = "float32" # Doesn't actually do anything since float32 is the default anyways. Only recognized other dtype is "bfloat16"

    if not "iterations" in params.keys():
        params["iterations"] = 1 # Because this controls how many samples are prefetched

    logger.info(params)

    model_fn = models[params["model"]][0]
    predict_fn = models[params["model"]][1]
    input_fn = inputs[params["input"]]

    if params["use_tpu"] and not predict_mode:
        # Resolve TPU cluster and runconfig
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=params["model_path"],
            cluster=tpu_cluster_resolver,
            save_checkpoints_secs=60*30,
            session_config=tf.ConfigProto(
                # allow_soft_placement=True,
                # log_device_placement=True
                ),
                tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=params["iterations"])
        )

        # Set up network
        network = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                use_tpu=True,
                train_batch_size=params["train_batch_size"], # These are the global sizes, must be divisible by replicas
                eval_batch_size=params["eval_batch_size"],
                predict_batch_size=params["predict_batch_size"],
                config=run_config,
                params=params)

    else:
        # Non TPU setup
        if not predict_mode:
            params["batch_size"] = params["train_batch_size"]
        else:
            params["batch_size"] = params["predict_batch_size"]

            from GPT2_models.models.gpt2 import encoder
            enc = encoder.get_encoder(params["encoder_path"])
            tokens = enc.encode(text)
            params["text_len"] = len(tokens)
            if params["text_len"] > 1024:
                params["text_len"] = 1024

        run_config = tf.estimator.RunConfig(
            model_dir=params["model_path"],
            session_config=tf.ConfigProto(
                # log_device_placement=True,
                # allow_soft_placement=True
            ),
        )

        network = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params)

    if predict_mode:
        logger.info("Generating predictions...")
        
        return predict_fn(network, text, params)
        #sys.exit()

    # Train eval loop
    while True:
        start = time.time()

        network.train(
                input_fn=partial(input_fn, eval=False),
                steps=params["train_steps"])


        end = time.time()
        logger.info("\nTrain loop took {:.2f}s\n".format(end-start))

        eval_result = network.evaluate(
           input_fn=partial(input_fn, eval=True),
           steps=params["eval_steps"])

        logger.info("\nEval Results: {}\n".format(str(eval_result)))

        if network.get_variable_value("global_step") > params["max_steps"]:
            logger.info("Done!")
            break
