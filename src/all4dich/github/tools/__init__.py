import requests
import argparse
import json
import logging


# arg_parser = argparse.ArgumentParser(description='GitHub tools')
# arg_parser.add_argument('-t', '--token', help='GitHub Personal Token', required=False)
# arg_parser.add_argument('-o', '--org', help='GitHub Organization', required=False)
# args = arg_parser.parse_args(

class org_manager:
    @staticmethod
    def get_org_user_list(org, token):
        req_url = "https://api.github.com/orgs/" + org + "/members?per_page=1000"
        req_headers = {'Authorization': 'token ' + token}
        logging.info(f"Requesting {req_url}")
        req = requests.get(req_url, headers=req_headers)
        user_lists = req.json()
        filtered_user_lists = []
        logging.info("Processing user lists")
        for each_user in user_lists:
            user_login = each_user['login']
            user_url = each_user['url']
            user_info_res  = requests.get(user_url, headers=req_headers)
            user_info = user_info_res.json()
            user_name = user_info['name']
            user_email = user_info['email']
            filtered_user_lists.append({'login': user_login, 'name': user_name, 'email': user_email})
        logging.info("Done processing user lists")
        return filtered_user_lists

