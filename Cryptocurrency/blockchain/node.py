import hashlib

class Node:
    """
    Node is a node of a merkletree and stores its left
    and right child nodes, his parent node and a hash 
    of a transaction or a hash of the hashes of its childs
    """
    def __init__(self, parent):
        """
        Constructor
        parent: Node
        """
        self.transaction_hash = None
        self.parent = parent
        self.lc = None
        self.rc = None
        
    def add_transaction(self, transaction):
        """
        stores the hash of a transaction in the Node

        Arguments:
        transaction: Transaction -- the transaction, whose hash need to be stored in node
        """
        self.transaction_hash = self.hash(transaction)

    def add_lc(self,lc):
        """
        adds a reference to his left Child node

        Arguments:
        lc: Node -- node, which is made to the left child of the current node 
        """
        self.lc = lc

    def add_rc(self,rc):
        """
        adds a reference to his right Child node

        Arguments:
        rc: Node -- node, which is made to the right child of the current node 
        """
        self.rc = rc

    def hash(self, value):
        """
        hashes a transaction

        Argument:
        value: Transaction -- transaction, which needs to be hashed

        Return:
        h.hexdigest: String
        """
        h = hashlib.sha256()
        h.update(str(value).encode('utf-8'))
        return h.hexdigest()
