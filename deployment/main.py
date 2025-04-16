import argparse
from typing import Dict
from kserve import Model, ModelServer, InferOutput, InferResponse
from kserve.utils.utils import generate_uuid
import spacy
from src.classifier import OrganizationSubclassifier

class MyModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False
        self.load()
        
    def load(self):
        self.model = spacy.load("saved_model/entity_classifier")
        self.ready = True
        
    # Override the method without changing its signature
    def preprocess(self, payload, headers=None):
        print("Preprocess called!")
        # Extract input data
        if isinstance(payload, dict):
            inputs = payload.get("inputs", [])
            if inputs and "data" in inputs[0]:
                input_data = inputs[0]["data"][0]
            else:
                input_data = ""
        else:
            # Assume it's an InferRequest
            input_data = payload.inputs[0].data[0]
        
        # Convert to string if needed
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')
            
        return input_data
        
    def predict(self, input_text, headers=None):
        print("Predict called!")
        doc = self.model(input_text)
        result = "UNKNOWN"
        
        if doc.ents:
            result = doc.ents[0].label_
            # Get subtype for organizations if available
            if result == "ORG" and hasattr(doc, '_') and hasattr(doc._, 'org_subtypes'):
                if input_text in doc._.org_subtypes:
                    result = doc._.org_subtypes[input_text]
        
       
        infer_output = InferOutput(name="output-0", shape=[1], datatype="BYTES", data=[result])
        return InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=generate_uuid())
        
    def postprocess(self, response, headers=None):
        print("Postprocess called!")
        print(response)
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="model", help="The name of the model")
    args, _ = parser.parse_known_args()
    
    model = MyModel(args.model_name)
    server = ModelServer()
    server.start([model])