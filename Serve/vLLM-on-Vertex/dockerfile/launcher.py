# python3 luanch.py --model_gcs_uri=gcs://bucket/base_model --peft_model_gcs_uri=gcs://bucket/peft_model
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--tensor-parallel-size", nargs='?', type=int)
    parser.add_argument("--model_gcs_uri", nargs='?', type=str)
    parser.add_argument("--peft_model_gcs_uri", nargs='?', type=str)

    return parser.parse_known_args()

def download_model(model_gcs_uri):
    print ("download model from gcs")
    model_path = "/gcs-mount/" + "/".join(model_gcs_uri.split("/")[3:])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    os.system("gsutil -m cp -r %s %s" % (model_gcs_uri + "/*", model_path))
    print ("download model completed")
    return model_path


def main():
    args, vllm_args = get_args()

    is_vertex_ai = os.environ.get("AIP_FRAMEWORK")

    if is_vertex_ai is not None:
        # the runtime is vertex ai custom serving environment
        print ("the runtime is vertex ai custom serving environment")

        if args.model_gcs_uri is not None:
            model_path = download_model(args.model_gcs_uri)
        if args.peft_model_gcs_uri is not None:
            peft_model_path = download_model(args.peft_model_gcs_uri)

    else:
        # the runtime is docker
        print ("the runtime is docker")
    
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
            result = os.system("python3 -m vllm.entrypoints.api_server --model='%s' %s" % (merged_model_path, " ".join(vllm_args)))
            print (result)
        else:
            # start vllm server
            print ("serve gcs model")
            result = os.system("python3 -m vllm.entrypoints.api_server  --model='%s' %s" % ( model_path, " ".join(vllm_args)))
            print (result)
    else:
        print ("you do not specify a model, the default model(facebook/opt-125m) will be used for serving.")

        result = os.system("python3 -m vllm.entrypoints.api_server  %s" % (" ".join(vllm_args)))
        print (result)


if __name__ == "__main__" :
    main()