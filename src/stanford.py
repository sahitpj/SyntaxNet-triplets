# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import requests

def treegex_api(patterns, text, url="http://localhost:9000/tregex"):
    pattern = None
    responeses = list()
    for p in patterns:
        pattern = p 
        request_params = {"pattern": p}
        r = requests.post(url, data=text, params=request_params)
        responeses.append(r.json())
    return responeses



# request_params = {"pattern": "(NP[$VP]>S)|(NP[$VP]>S\\n)|(NP\\n[$VP]>S)|(NP\\n[$VP]>S\\n)"}
