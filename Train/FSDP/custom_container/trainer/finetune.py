import fire                                                                   
from llama_recipes.finetuning import main                                          
from google.cloud import logging_v2                                                
import sys                                                                        
import os                                                                          
from google.cloud.logging_v2.handlers import setup_logging                        
from google.cloud.logging.handlers import CloudLoggingHandler                                                                                  
from google.cloud import logging as google_logging                                 
import logging                                                                                                                                                                                                                                                     
# Get job id from env CLOUD_ML_JOB_ID                                                                                    
def GetJobID():
    job_id = os.getenv("CLOUD_ML_JOB_ID")
    return job_id                                                                                                                                                                                                                                                       
# Get project number from env CLOUD_ML_PROJECT_ID
def GetProjectNumber():
    project_id = os.getenv("CLOUD_ML_PROJECT_ID")
    return project_id

# Get node id from env CLUSTER_SPEC
def GetNodeInfor(worker_pool_spec: str = None):
    if worker_pool_spec == None:
        cluster_spec_str = os.getenv("CLUSTER_SPEC")
        cluster_spec_dict = json.loads(cluster_spec_str)
    else:
        cluster_spec_dict = json.loads(worker_pool_spec)
    
    node_pool = cluster_spec_dict["task"]["type"]
    node_index = cluster_spec_dict["task"]["index"]
    node_id = cluster_spec_dict["cluster"][node_pool][int(node_index)]
    return node_id

if __name__ == "__main__":
    logging_client = google_logging.Client(project=GetProjectNumber())
    log_name = "ml.googleapis.com%2F" + GetJobID()
    resource = logging_v2.Resource(type="ml_job", labels={"job_id": GetJobID(), "task_name": "service", "project_id": GetProjectNumber()})
    handler = CloudLoggingHandler(client=logging_client, resource=resource, name=log_name, labels={"node_id": GetNodeInfor()})
    setup_logging(handler)
    
    sys.stderr.write = logging.error
    sys.stdout.write = logging.info
    
    fire.Fire(main)  