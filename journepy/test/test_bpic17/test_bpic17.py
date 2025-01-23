import pyrootutils
path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(
path=path, # path to the root directory
project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
dotenv=True, # load environment variables from .env if exists in root directory
pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
cwd=True, # change current working directory to the root directory (helps with filepaths)
)
import unittest
import networkx as nx
from pm4py.objects.log.importer.xes import importer as xes_importer
from src.preprocessing import bpic17, preprocessing_utils
from src import game_construction, game_operations
import main_bpi2017

UPPAAL_STRATEGO = "" # set UPPAAL Strateo path here

class TestBPC17Methods(unittest.TestCase):
    #def setUp(self):
        # load procesed event log
        # print("start")
    
    def test_game_construction(self):
        log_before, log_after = bpic17.preprocessed_log("data/orig/BPI Challenge 2017.xes")
        g_before = game_construction.game(log_before, 3, game_construction.sequence, actor_path = "data/activities2017.xml")
        g_before = main_bpi2017.color_graph(g_before)
        g_after = game_construction.game(log_after, 3, game_construction.sequence, actor_path = "data/activities2017.xml")
        g_after = main_bpi2017.color_graph(g_after)

        g_before_stored = nx.read_gexf("test/test_bpic17/res/graph_before.gexf")
        g_after_stored = nx.read_gexf("test/test_bpic17/res/graph_after.gexf")

        # check on crucial elements : action, cost, controllable and color
        # note that notebook calls cost "label"
        self.assertSetEqual(set(g_before.nodes), set(g_before_stored.nodes))
        self.assertSetEqual(set(g_before.edges), set(g_before_stored.edges))
        for e in g_before.edges:
            self.assertTrue(g_before.edges[e]['action'] == g_before_stored.edges[e]['action'])
            self.assertTrue(g_before.edges[e]['cost'] == g_before_stored.edges[e]['cost'])
            self.assertTrue(g_before.edges[e]['controllable'] == g_before_stored.edges[e]['controllable'])
            self.assertTrue(g_before.edges[e]['color'] == g_before_stored.edges[e]['color'])

        self.assertSetEqual(set(g_after.nodes), set(g_after_stored.nodes))
        self.assertSetEqual(set(g_after.edges), set(g_after_stored.edges))
        for e in g_after.edges:
            self.assertTrue(g_after.edges[e]['action'] == g_after_stored.edges[e]['action'])
            self.assertTrue(g_after.edges[e]['cost'] == g_after_stored.edges[e]['cost'])
            self.assertTrue(g_after.edges[e]['controllable'] == g_after_stored.edges[e]['controllable'])
            self.assertTrue(g_after.edges[e]['color'] == g_after_stored.edges[e]['color'])
    
    def test_unrolled_model(self):
        g_before_stored = nx.read_gexf("test/test_bpic17/res/graph_before.gexf")
        g_after_stored = nx.read_gexf("test/test_bpic17/res/graph_after.gexf")
        target = [s for s in g_before_stored.nodes if "positive" in s or "negative" in s]
        g_before_unroll = game_operations.unroll(g_before_stored, "start", target, 1)
        target = [s for s in g_after_stored.nodes if "positive" in s or "negative" in s]
        g_after_unroll = game_operations.unroll(g_after_stored, "start", target, 1)
        g_before_unrolled_stored = nx.read_gexf("test/test_bpic17/res/graph_before_unrolled.gexf")
        g_after_unrolled_stored = nx.read_gexf("test/test_bpic17/res/graph_after_unrolled.gexf")
        nx.write_gexf(g_before_unroll, "test/test_bpic17/res/graph_before_unrolled.gexf")
        nx.write_gexf(g_after_unroll, "test/test_bpic17/res/graph_after_unrolled.gexf")

        # check on crucial elements : action, cost, controllable and color
        # note that notebook calls cost "label"
        self.assertSetEqual(set(g_before_unroll.nodes), set(g_before_unrolled_stored.nodes))
        self.assertSetEqual(set(g_before_unroll.edges), set(g_before_unrolled_stored.edges))
        for e in g_before_unroll.edges:
            self.assertTrue(g_before_unroll.edges[e]['action'] == g_before_unrolled_stored.edges[e]['action'])
            self.assertTrue(g_before_unroll.edges[e]['cost'] == g_before_unrolled_stored.edges[e]['label'])
            self.assertTrue(g_before_unroll.edges[e]['controllable'] == g_before_unrolled_stored.edges[e]['controllable'])
            self.assertTrue(g_before_unroll.edges[e]['color'] == g_before_unrolled_stored.edges[e]['color'])

        self.assertSetEqual(set(g_after_unroll.nodes), set(g_after_unrolled_stored.nodes))
        self.assertSetEqual(set(g_after_unroll.edges), set(g_after_unrolled_stored.edges))
        for e in g_after_unroll.edges:
            self.assertTrue(g_after_unroll.edges[e]['action'] == g_after_unrolled_stored.edges[e]['action'])
            self.assertTrue(g_after_unroll.edges[e]['cost'] == g_after_unrolled_stored.edges[e]['label'])
            self.assertTrue(g_after_unroll.edges[e]['controllable'] == g_after_unrolled_stored.edges[e]['controllable'])
            self.assertTrue(g_after_unroll.edges[e]['color'] == g_after_unrolled_stored.edges[e]['color'])
    
    def test_unrolling_1(self):
        start = "1"
        G=nx.DiGraph()
        G.add_nodes_from(["1","2","3","4"])
        G.add_edges_from([("1","2"),("1","3"),("2","3"),("3", "2"), ("2", "4")])
        target = "4"
        G_gen = game_operations.unroll(G, start,[target],1)
        g_target = nx.DiGraph()
        g_target.add_edges_from([('1', '2'), ('1', '3'), ('2', '3.1'), ('2', '4'), ('3', '2.1'), ('3.1', '2.2'), ('2.1', '3.2'), ('2.1', '4'), ('2.2', '4'), ('3.2', '2.3'), ('2.3', '4')])
        self.assertTrue(nx.is_isomorphic(G_gen, g_target))

    def test_unrolling_0(self):
        start = "1"
        G=nx.DiGraph()
        G.add_nodes_from(["1","2","3","p", "n"])
        G.add_edges_from([("1","2"),("2","1"),("1","3"),("3","1"),("2","3"),("3", "2"), ("1", "n"),("3", "p")])
        target = ['p', 'n']
        G_gen = game_operations.unroll(G, start,target,0)
        g_target = nx.DiGraph()
        g_target.add_edges_from([('1', '2'), ('1', '3'), ('1', 'n'), ('2', '3.1'), ('3', 'p'), ('3.1', 'p')])
        self.assertTrue(nx.is_isomorphic(G_gen, g_target))
    
    def test_decision_boundary_and_reduction(self):
        g_before_stored = nx.read_gexf("test/test_bpic17/res/graph_before.gexf")
        db_game_before, db_before = game_operations.compute_db(g_before_stored, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./")
        g_after_stored = nx.read_gexf("test/test_bpic17/res/graph_after.gexf")
        db_game_after, db_after = game_operations.compute_db(g_after_stored, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./")

        # assure that decision boundary is equal
        self.assertSetEqual(set(db_before), set(['A_Validating - O_Returned - W_Call incomplete files SHORT', 'A_Validating - O_Returned - W_Call incomplete files LONG', 'A_Accepted - O_Create Offer 0 - O_Sent (online only)', 'W_Call after offers SHORT - O_Create Offer 1 - O_Sent (online only)']))
        self.assertSetEqual(set(db_after), set(['A_Validating - O_Returned - W_Call incomplete files SHORT', 'A_Validating - O_Returned - W_Call incomplete files LONG', 'W_Call after offers SHORT - O_Create Offer 1 - O_Sent (online only)', 'O_Sent (mail and online) - O_Create Offer 1 - O_Sent (online only)', 'W_Call incomplete files SHORT - O_Returned - W_Call incomplete files SHORT']))

        reduced_db_game_before = game_operations.db_reduction(db_game_before)
        reduced_db_game_after = game_operations.db_reduction(db_game_after)

        reduced_db_game_before_stored = nx.read_gexf("test/test_bpic17/res/reduced_db_game_before.gexf")
        reduced_db_game_after_stored = nx.read_gexf("test/test_bpic17/res/reduced_db_game_after.gexf")    

        self.assertSetEqual(set(reduced_db_game_before.nodes), set(reduced_db_game_before_stored.nodes))
        self.assertSetEqual(set(reduced_db_game_before.edges), set(reduced_db_game_before_stored.edges))
        for e in reduced_db_game_before.edges:
            self.assertTrue(reduced_db_game_before.edges[e]['action'] == reduced_db_game_before_stored.edges[e]['action'])
            self.assertAlmostEqual(reduced_db_game_before.edges[e]['cost'], reduced_db_game_before_stored.edges[e]['cost'])
            self.assertTrue(reduced_db_game_before.edges[e]['controllable'] == reduced_db_game_before_stored.edges[e]['controllable'])
            self.assertTrue(reduced_db_game_before.edges[e]['color'] == reduced_db_game_before_stored.edges[e]['color'])

        self.assertSetEqual(set(reduced_db_game_after.nodes), set(reduced_db_game_after_stored.nodes))
        self.assertSetEqual(set(reduced_db_game_after.edges), set(reduced_db_game_after_stored.edges))
        for e in reduced_db_game_after.edges:
            self.assertTrue(reduced_db_game_after.edges[e]['action'] == reduced_db_game_after_stored.edges[e]['action'])
            self.assertTrue(reduced_db_game_after.edges[e]['cost'] == reduced_db_game_after_stored.edges[e]['cost'])
            self.assertTrue(reduced_db_game_after.edges[e]['controllable'] == reduced_db_game_after_stored.edges[e]['controllable'])
            self.assertTrue(reduced_db_game_after.edges[e]['color'] == reduced_db_game_after_stored.edges[e]['color'])


    def test_static_decision_boundary(self):
        g_before_stored = nx.read_gexf("test/test_bpic17/res/graph_before.gexf")
        g_after_stored = nx.read_gexf("test/test_bpic17/res/graph_after.gexf")
        db_game_before_static, db_before_static = game_operations.compute_db(g_before_stored, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./", static=True)
        db_game_after_static, db_after_static = game_operations.compute_db(g_after_stored, "guaranteed_tool.q", uppaal_stratego=UPPAAL_STRATEGO, output = "./", static=True)
        reduced_db_game_before_static = game_operations.db_reduction(db_game_before_static, static=True)
        reduced_db_game_after_static = game_operations.db_reduction(db_game_after_static, static=True)

        # assure that decision boundary is equal
        self.assertSetEqual(set(db_before_static), set(['A_Validating - O_Returned - W_Call incomplete files SHORT', 'A_Validating - O_Returned - W_Call incomplete files LONG', 'A_Accepted - O_Create Offer 0 - O_Sent (online only)', 'W_Call after offers SHORT - O_Create Offer 1 - O_Sent (online only)']))
        self.assertSetEqual(set(db_after_static), set(['A_Validating - O_Returned - W_Call incomplete files SHORT', 'A_Validating - O_Returned - W_Call incomplete files LONG', 'W_Call after offers SHORT - O_Create Offer 1 - O_Sent (online only)', 'O_Sent (mail and online) - O_Create Offer 1 - O_Sent (online only)', 'W_Call incomplete files SHORT - O_Returned - W_Call incomplete files SHORT']))

        reduced_db_game_before_stored_static = nx.read_gexf("test/test_bpic17/res/reduced_db_game_before_static.gexf")
        reduced_db_game_after_stored_static = nx.read_gexf("test/test_bpic17/res/reduced_db_game_after_static.gexf")    

        self.assertSetEqual(set(reduced_db_game_before_static.nodes), set(reduced_db_game_before_stored_static.nodes))
        self.assertSetEqual(set(reduced_db_game_before_static.edges), set(reduced_db_game_before_stored_static.edges))
        for e in reduced_db_game_before_static.edges:
            self.assertTrue(reduced_db_game_before_static.edges[e]['action'] == reduced_db_game_before_stored_static.edges[e]['action'])
            self.assertAlmostEqual(reduced_db_game_before_static.edges[e]['cost'], reduced_db_game_before_stored_static.edges[e]['cost'])
            self.assertTrue(reduced_db_game_before_static.edges[e]['controllable'] == reduced_db_game_before_stored_static.edges[e]['controllable'])
            self.assertTrue(reduced_db_game_before_static.edges[e]['color'] == reduced_db_game_before_stored_static.edges[e]['color'])

        self.assertSetEqual(set(reduced_db_game_after_static.nodes), set(reduced_db_game_after_stored_static.nodes))
        self.assertSetEqual(set(reduced_db_game_after_static.edges), set(reduced_db_game_after_stored_static.edges))
        for e in reduced_db_game_after_static.edges:
            self.assertTrue(reduced_db_game_after_static.edges[e]['action'] == reduced_db_game_after_stored_static.edges[e]['action'])
            self.assertTrue(reduced_db_game_after_static.edges[e]['cost'] == reduced_db_game_after_stored_static.edges[e]['cost'])
            self.assertTrue(reduced_db_game_after_static.edges[e]['controllable'] == reduced_db_game_after_stored_static.edges[e]['controllable'])
            self.assertTrue(reduced_db_game_after_static.edges[e]['color'] == reduced_db_game_after_stored_static.edges[e]['color'])

if __name__ == '__main__':
    unittest.main()

