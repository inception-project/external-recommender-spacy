import base64
from collections import namedtuple
from typing import Any, Dict

from flask import Flask, request, jsonify

from cassis import *

import spacy
from spacy.tokens import Doc

# Types

JsonDict = Dict[str, Any]

PredictionRequest = namedtuple("PredictionRequest", ["layer", "feature", "projectId", "document", "typeSystem"])
PredictionResponse = namedtuple("PredictionResponse", ["document"])
Document = namedtuple("Document", ["xmi", "documentId", "userId"])

# Constants

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"

# Models

nlp = spacy.load('en', disable=['parser'])

# Routes

app = Flask(__name__)


@app.route("/ner/predict", methods=["POST"])
def route_predict_ner():
    json_data = request.get_json()

    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_ner(prediction_request)

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/ner/train", methods=["POST"])
def route_train_ner():
    json_data = request.get_json()
    import json
    with open('req.json', 'w') as f:
        json.dump(json_data, f)
    return ('', 204)


@app.route("/pos/predict", methods=["POST"])
def route_predict_pos():
    json_data = request.get_json()

    prediction_request = parse_prediction_request(json_data)
    prediction_response = predict_pos(prediction_request)

    result = jsonify(document=prediction_response.document)

    return result


@app.route("/pos/train", methods=["POST"])
def route_train_pos():
    # Return empty response
    return ('', 204)


def parse_prediction_request(json_object: JsonDict) -> PredictionRequest:
    metadata = json_object["metadata"]
    document = json_object["document"]

    layer = metadata["layer"]
    feature = metadata["feature"]
    projectId = metadata["projectId"]

    xmi = document["xmi"]
    documentId = document["documentId"]
    userId = document["userId"]
    typesystem = json_object["typeSystem"]

    return PredictionRequest(layer, feature, projectId, Document(xmi, documentId, userId), typesystem)


# NLP

def predict_ner(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    # Extract the tokens from the CAS and create a spacy doc from it
    tokens = list(cas.select(TOKEN_TYPE))
    words = [cas.get_covered_text(token) for token in tokens]
    doc = Doc(nlp.vocab, words=words)

    # Find the named entities
    nlp.entity(doc)

    # For every entity returned by spacy, create an annotation in the CAS
    for ent in doc.ents:
        fields = {'begin': tokens[ent.start].begin,
                  'end': tokens[ent.end - 1].end,
                  prediction_request.feature: ent.label_}
        annotation = AnnotationType(**fields)
        cas.add_annotation(annotation)

    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


def predict_pos(prediction_request: PredictionRequest) -> PredictionResponse:
    # Load the CAS and type system from the request
    typesystem = load_typesystem(prediction_request.typeSystem)
    cas = load_cas_from_xmi(prediction_request.document.xmi, typesystem=typesystem)
    AnnotationType = typesystem.get_type(prediction_request.layer)

    # Extract the tokens from the CAS and create a spacy doc from it
    tokens = list(cas.select(TOKEN_TYPE))
    words = [cas.get_covered_text(token) for token in tokens]
    doc = Doc(nlp.vocab, words=words)

    # Do the tagging
    nlp.tagger(doc)

    # For every token, extract the POS tag and create an annotation in the CAS
    for token in doc:
        fields = {'begin': tokens[token.i].begin,
                  'end': tokens[token.i].end,
                  prediction_request.feature: token.pos_}
        annotation = AnnotationType(**fields)
        cas.add_annotation(annotation)

    xmi = cas.to_xmi()
    return PredictionResponse(xmi)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    """
    # For debugging purposes, load a json file containing the request and process it.
    import json
    with open("predict.json", "rb") as f:
        predict_json = json.load(f)

    request = parse_prediction_request(predict_json)
    predict_pos(request)
    """
