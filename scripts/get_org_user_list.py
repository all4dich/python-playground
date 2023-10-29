import argparse
import csv
from all4dich.github.tools import org_manager
import tempfile

arg_parser = argparse.ArgumentParser(description='GitHub tools')
arg_parser.add_argument('-t', '--token', help='GitHub Personal Token', required=False)
arg_parser.add_argument('-o', '--org', help='GitHub Organization', required=False)
args = arg_parser.parse_args()

# Set console log level as INFO
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    r = org_manager.get_org_user_list(args.org, args.token)
    header_names = ['login', 'name', 'email']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_names)
        writer.writeheader()
        for each_user in r:
            writer.writerow(each_user)
    print(f"Output file = {csvfile.name}")
