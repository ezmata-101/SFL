GOOGLE_BUCKET_NAME = 'sfl-data'

SERVER_PRIVATE_ADDRESS = '10.148.0.2'
SERVER_PUBLIC_ADDRESS = '34.142.168.86'

CLIENT_PUBLIC_ADDRESSES = [
    '35.197.130.4',
    '34.87.157.131',
]

CLIENT_PRIVATE_ADDRESSES = [
    '10.148.0.5',
    '10.148.0.6'
]

TOTAL_CLIENTS = 2

COMMON_USERNAME = 'sflu'


def GET_CLIENT_ID():
    return int(CLIENT_PRIVATE_ADDRESSES[0].split('.')[-1]) - 5



