import secrets
from ecdsa import SigningKey, SECP256k1
import hashlib
import base58
import requests
import argparse


def generate_private_key():
    return secrets.token_hex(nbytes=32)


def generate_public_key(private_key):
    bytes_pk = bytes.fromhex(private_key)
    signing_key = SigningKey.from_string(bytes_pk, curve=SECP256k1)
    verifying_key = signing_key.verifying_key.to_string()
    return b'\x04' + verifying_key


def generate_address(public_key):
    ripemd_160 = hashlib.new('ripemd160')
    hash_sha256 = hashlib.sha256(public_key).digest()
    ripemd_160.update(hash_sha256)
    hash_ripemd_160 = ripemd_160.digest()
    interim = b'\x00' + hash_ripemd_160
    hash2_sha256 = hashlib.sha256(interim).digest()
    hash3_sha256 = hashlib.sha256(hash2_sha256).digest()
    first_four_bytes = hash3_sha256[0:4]
    address_bytes = base58.b58encode(interim + first_four_bytes)
    return address_bytes.decode('utf-8')


def request_balance(api_key, address):
    URL = 'https://www.blockonomics.co/api/balance'
    headers = {'Authorization': "Bearer " + api_key}
    r = requests.post(URL, headers=headers, json={'addr':address})
    response = r.json()
    balance = response['response'][0]['confirmed']
    return balance


def main(api_key):
    pk = generate_private_key()
    pubk = generate_public_key(pk)
    address = generate_address(pubk)
    balance = request_balance(api_key, address)
    names = ['Address', 'Balance BTC', 'Private Key', 'Public Key']
    results = dict(zip(names,(address, str(balance), pk, pubk.hex())))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('key', help='apikey for Blockonomics')
    args = parser.parse_args()
    results = main(args.key)
    for key, value in results.items():
        print(key,': ', value)
