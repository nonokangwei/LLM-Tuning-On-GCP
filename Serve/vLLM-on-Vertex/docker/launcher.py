# python3 luanch.py --model_gcs_uri=gcs://bucket/base_model --peft_model_gcs_uri=gcs://bucket/peft_model


import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", nargs='?', type=int)
    parser.add_argument("--model_gcs_uri", nargs='?', type=str)
    parser.add_argument("--peft_model_gcs_uri", nargs='?', type=str)

    return parser.parse_args()

def download_model(model_gcs_uri):
    print ("download model from gcs")
    model_path = "/gcs-mount/" + "/".join(model_gcs_uri.split("/")[3:])
    os.system("gsutil cp %s %s" % (model_gcs_uri, model_path))
    print ("download model completed")
    return model_path


def main():
    args = get_args()

    is_vertex_ai = os.path.exists("AIP_FRAMEWORK")

    if is_vertex_ai is not None:
        # the runtime is vertex ai custom serving environment
        print ("the runtime is vertex ai custom serving environment")
        serve_port = 7080
    else:
        # the runtime is docker
        print ("the runtime is docker")
        serve_port = 8000

    
    if args.model_gcs_uri is not None:
        print ("load your model from gcs")
        model_path = "/gcs-mount/" + "/".join(args.model_gcs_uri.split("/")[3:])

        if args.peft_model_gcs_uri is not None:
            print ("load your peft model from gcs")
            peft_model_path = "/gcs-mount/" + "/".join(args.peft_model_gcs_uri.split("/")[3:])

            # merge base model and peft model
            print ("start merging base model and peft model")
            merged_model_path = "/gcs-mount/peft_merged_model"
            os.system("python3 /root/scripts/merge_peft.py --base_model='%s' --peft_model='%s' --saved_path='%s'" % (model_path, peft_model_path, merged_model_path))
            print ("base model and peft model merge completed")
            
            # start vllm server
            print ("serve gcs peft fine tuned model")
            result = os.system("python3 -m vllm.entrypoints.api_server --tensor-parallel-size='%s' --model='%s' --host=0.0.0.0 --port='%s'" % (args.tensor_parallel_size, merged_model_path, str(serve_port)))
            print (result)
        else:
            # start vllm server
            print ("serve gcs model")
            result = os.system("python3 -m vllm.entrypoints.api_server  --tensor-parallel-size='%s' --model='%s' --host=0.0.0.0 --port='%s'" % (args.tensor_parallel_size, merged_model_path, str(serve_port)))
            print (result)
    else:
        print ("you do not specify a model, the default model(facebook/opt-125m) will be used for serving.")

        result = os.system("python3 -m vllm.entrypoints.api_server  --tensor-parallel-size='%s' --host=0.0.0.0 --port='%s'" % (args.tensor_parallel_size, str(serve_port)))
        print (result)


if __name__ == "__main__" :
    main()