from datetime import datetime
import json
import hashlib
import uuid
from flask import Flask, request


class Block:
    def __init__(self, transactions, previous_hash):
        self.id = str(uuid.uuid4())
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = str(datetime.now())

    def compute_hash(self):
        return hashlib.sha224(json.dumps(self.__dict__).encode('utf-8')).hexdigest()


class BlockChain:
    pow_difficulty = 3

    def __init__(self):
        self.pending_transactions = []
        self.chain = []
        self.create_genesis_block()

    @property
    def last_block(self):
        return self.chain[-1]

    def create_genesis_block(self):
        """Creates genesis block, function called when instance created"""
        genesis_block = Block(transactions=[], previous_hash='genesis')
        self.chain.append(genesis_block)
        return

    def proof_of_work(self, block):
        """Returns hash for the block that meets the proof of work"""
        block.nonce = 0
        proof = block.compute_hash()
        while not proof.startswith(self.pow_difficulty * '0'):
            block.nonce += 1
            proof = block.compute_hash()
        return proof

    def verify_block(self, block, proof):
        """Verify proposed proof equals block_hash and proof of work constraint met"""
        return (block.compute_hash() == proof) and proof.startswith(self.pow_difficulty * '0')

    def add_block(self, block):
        """Adds block to chain"""
        proof = self.proof_of_work(block)
        verified = self.verify_block(block, proof)
        if verified:
            self.chain.append(block)
            self.pending_transactions.clear()
            return True
        else:
            return False

    def mine(self):
        """Mine pending transactions if there are any"""
        last_block = self.last_block
        block = Block(transactions=self.pending_transactions.copy(),
                      previous_hash=last_block.compute_hash())
        block_added = self.add_block(block)
        return block_added


blockchain = BlockChain()
app = Flask(__name__)
nodes = set()


@app.route('/mine')
def mine_pending_transactions():
    if not blockchain.pending_transactions:
        return 'No pending transactions to mine'
    else:
        mining = blockchain.mine()
        return blockchain.last_block.id if mining else 'Block could not be verified'


@app.route('/add_transactions', methods=['POST'])
def add_transactions():
    transactions = dict(request.json)
    blockchain.pending_transactions.append(transactions)
    return 'Pending transactions added'


@app.route('/add_node')
def add_node():
    print(request.url)
    print(request.host_url)
    return 'Success1'

