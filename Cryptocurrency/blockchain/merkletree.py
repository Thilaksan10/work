import hashlib
import math
from blockchain.node import Node


class Merkletree:

    def __init__(self, transactions):
        """
        Constructor
        transactions: List<Transaytion>
        """
        transactions_amount = len(transactions)
        if transactions_amount == 0:
            target_height = -1
        else:
            target_height = math.log2(transactions_amount)
        self.root = self.create_tree(Node(None), target_height, transactions, 0)
            
    def create_tree(self, root, target_height, transactions, transaction_no):
        """
        creates a merkle tree and stores the hashes of the hash pairs in the 
        node it belongs to

        Arguments:
        root: Node -- current node, on which the method is operating
        target_height: int -- the height the ,´merkle tree needs to have
        transactions: List<Transaction> -- transactions, which are gonna get hashed and 
                                            stored in the lowest level of the merkle tree
        transaction_no: int -- points on a transaction, which has to be hashed and 
                                stored next in the merkle tree 

        Return: 
        root: Node
        self.create_tree: Node
        """
        if root.parent == None and root.lc != None and root.rc != None:
            root.transaction_hash = self.hash(root.lc.transaction_hash, root.rc.transaction_hash)
            return root
        elif len(transactions) == 1:
            root.add_transaction(transactions[transaction_no])
            return root
        elif target_height == -1:
            root.transaction_hash = '0'*64
            return root
        elif root.lc == None and target_height > 0:
            lc = Node(root)
            root.add_lc(lc)
            return self.create_tree(lc,target_height - 1, transactions, transaction_no)
        elif root.rc == None and target_height > 0:
            rc = Node(root)
            root.add_rc(rc)
            return self.create_tree(rc, target_height - 1, transactions, transaction_no)
        elif target_height == 0:
            root.add_transaction(transactions[transaction_no])
            return self.create_tree(root.parent, target_height + 1, transactions, transaction_no+1)
        else:
            root.transaction_hash = self.hash(root.lc.transaction_hash, root.rc.transaction_hash)
            return self.create_tree(root.parent, target_height + 1, transactions, transaction_no)

    def hash(self, hash_a, hash_b):
        """
        hashes the hash, which are stored in both childs of the node

        Arguments:
        hash_a: String -- hash stored in left child
        hash_b: String -- hash stored in right child

        Return:
        h.hexdigest: String
        """
        h = hashlib.sha256()
        h.update(
            str(hash_a).encode('utf-8') +
            str(hash_b).encode('utf-8')
        )
        return h.hexdigest()

