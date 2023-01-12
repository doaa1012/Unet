import argparse as ag
import json


def get_parser_with_args(metadata_json='./metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    file = open(metadata_json, 'r', encoding = 'utf-8')
    metadata = file
    parser.set_defaults(**metadata)
    file.close()
    return parser, metadata


m, d = get_parser_with_args()
"""

def get_parser_with_args(metadata_json='metadata.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata

    return None

"""