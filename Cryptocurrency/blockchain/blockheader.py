import datetime
import hashlib

class Blockheader:
    """
    Blockheader stores the version number of the 
    software, the hash of the previous block,
    the root hash of the Merkle tree, the goal of
    the current difficulty, the time the Block 
    was created and the nonce, which stores the
    tries of the miner to create the block.
    """
    def __init__(self,version, previous_hash, merkle_root, difficulty, nonce):
        """
        Constructor 
        verion: int
        previous_hash: String
        merkle_root: String
        difficulty: ?
        nonce: int
        """
        self.version = version
        self.previous_hash = previous_hash
        self.merkle_root = merkle_root
        self.timeStamp = datetime.datetime.now()
        self.difficulty = difficulty
        self.nonce = nonce

    def hash(self):
        """
        creates the hash of the Block using the version 
        of the block, the hash of the previous block,
        ...,...
        and the nonce. 
        """
        h = hashlib.sha256()
        h.update(
            str(self.version).encode('utf-8') +
            str(self.previous_hash.encode).encode('utf-8') + 
            str(self.nonce).encode('utf-8') + 
            str(self.timeStamp).encode('utf-8')
        )
        return h.hexdigest()
    
    def __str__(self):
        return 'Version: ' + str(self.version) + '\nPrevious Hash: ' + str(self.previous_hash)  + '\nMerkle Root: ' + str(self.merkle_root) + '\nTime:' + str(self.timeStamp) + '\nNonce: ' + str(self.nonce)
