from blockchain.merkletree import Merkletree
from blockchain.blockheader import Blockheader
from blockchain.blockbody import Blockbody
import hashlib
import sys

class Block:
    """
    Block stores his magic number, which shows 
    that this is a Marecoin block, his blockheight
    his blocksize, blockheader and his blockbody, 
    with which it creates its own hash, which is 
    also stored in the block itself.
    """
    def __init__(self, blockheight, version, previous_hash, difficulty, nonce, transactions, mining_reward):
        """
        Constructor
        blockheader: Blockheader
        blockbody: Blockbody
        """
        self.software = 0x8F8A3808
        self.blockheight = blockheight
        merkle_tree = Merkletree(transactions)
        self.blockheader = Blockheader(version, previous_hash, merkle_tree.root.transaction_hash, difficulty, nonce)
        self.blockbody = Blockbody(transactions)
        self.own_hash = self.hash()
        self.blocksize = sys.getsizeof(self.software) + sys.getsizeof(self.blockheight) + sys.getsizeof(self.blockheader) + sys.getsizeof(self.blockbody) + sys.getsizeof(self.own_hash)
        self.accept_count = 0
        self.mining_reward = mining_reward

    def hash(self):
        h = hashlib.sha256()
        h.update(
            str(self.blockheader.hash()).encode('utf-8')
        )
        return h.hexdigest()

    def increase_accept_count(self):
        self.accept_count += 1
    
    def __str__(self):
        if self.blockheight == 0:
            return 'Software: ' + str(self.software) + '\nHeight: ' + str(self.blockheight) + '\nSize: ' + str(self.blocksize) + '\nHeader:\n' + str(self.blockheader) + '\nBody:\n' + str(self.blockbody) + '\nHash: ' + str(self.own_hash) + '\nMining Reward: ' +  str(self.mining_reward)
        else:
             return 'Software: ' + str(self.software) + '\nHeight: ' + str(self.blockheight) + '\nSize: ' + str(self.blocksize) + '\nHeader:\n' + str(self.blockheader) + '\nBody:\n' + str(self.blockbody) + '\nHash: ' + str(self.own_hash) + '\nMining Reward: ' +  str(self.mining_reward) + '\nTotal Reward: ' + str(self.blockbody.transactions[0].transaction_amount)
