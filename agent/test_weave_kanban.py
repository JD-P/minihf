import unittest
from typing import List, Optional, Dict, Any
import json
import types

# Assuming the classes WeaveKanbanTask and WeaveKanban are defined in a module named kanban
from weave_agent import WeaveAgent, WeaveKanbanTask, WeaveKanban

global agent

class TestWeaveKanbanTask(unittest.TestCase):

    def setUp(self):
        self.agent = WeaveAgent("placeholder")
        self.kanban = WeaveKanban(self.agent)

    def test_create_and_complete_task(self):
        task = WeaveKanbanTask(self.kanban, 1, "Test Task")
        task.completed("Task completed")
        self.assertEqual(task.status, 'completed')

    def test_create_task_with_evaluations_and_complete(self):
        def evaluation_callback(agent):
            return True

        task = WeaveKanbanTask(self.kanban, 1, "Test Task")
        task.add_evaluation("Test Evaluation", evaluation_callback)
        task.completed("Task completed")
        self.assertEqual(task.status, 'completed')

    def test_create_task_with_failing_evaluations(self):
        def evaluation_callback(agent):
            raise ValueError("Test Error")

        task = WeaveKanbanTask(self.kanban, 1, "Test Task")
        task.add_evaluation("Test Evaluation", evaluation_callback)
        with self.assertRaises(ValueError):
            task.completed("Task completed")

    def test_create_task_with_blockers_as_strings(self):
        with self.assertRaises(ValueError):
            WeaveKanbanTask(self.kanban, 1, "Test Task", status="blocked", blocked_on=["1"])

    def test_unblock_task_when_blockers_completed(self):
        task1 = WeaveKanbanTask(self.kanban, 1, "Blocker Task")
        self.kanban.tasks.append(task1)
        task2 = WeaveKanbanTask(self.kanban, 2, "Blocked Task", status="blocked", blocked_on=[1])
        self.kanban.tasks.append(task2)
        task1.completed("Blocker task completed")
        self.kanban.unblock()
        self.assertEqual(task2.status, 'idle')

    def test_str_id_becomes_int(self):
        task1 = WeaveKanbanTask(self.kanban, "1", "Test Task")  # task_id should be int
        self.assertEqual(task1.id, 1)
        task2 = WeaveKanbanTask(self.kanban, 1, 123)  # title should be str
        self.assertEqual(task2.title, "123")
        
    def test_create_task_with_wrong_arguments(self):
        with self.assertRaises(TypeError):
            # lambda cannot be converted to int
            WeaveKanbanTask(self.kanban, lambda x: x+1, "Test Task", status="invalid_status")
        class NoString:
            def __str__(self):
                pass
        with self.assertRaises(TypeError):
            WeaveKanbanTask(self.kanban, 1, NoString())  # nostring is invalid title
        with self.assertRaises(ValueError):
            WeaveKanbanTask(self.kanban, 1, "Test Task", status="invalid_status")  # invalid status

class TestWeaveKanban(unittest.TestCase):

    def setUp(self):
        self.agent = WeaveAgent("placeholder")
        self.kanban = WeaveKanban(self.agent)

    def test_add_task(self):
        self.kanban.add_task("Test Task")
        self.assertEqual(len(self.kanban.tasks), 1)
        self.assertEqual(self.kanban.tasks[0].title, "Test Task")

    def test_get_task(self):
        self.kanban.add_task("Test Task")
        task = self.kanban.get_task(1)
        self.assertIsNotNone(task)
        self.assertEqual(task.title, "Test Task")

    def test_view_board(self):
        self.kanban.add_task("Test Task 1")
        self.kanban.add_task("Test Task 2")
        board_view = self.kanban.view_board()
        self.assertIn("Test Task 1", board_view)
        self.assertIn("Test Task 2", board_view)

    def test_unblock(self):
        self.kanban.add_task("Blocker Task")
        self.kanban.add_task("Blocked Task", status="blocked", blocked_on=[1])
        blocker_task = self.kanban.get_task(1)
        blocked_task = self.kanban.get_task(2)
        blocker_task.completed("Blocker task completed")
        self.kanban.unblock()
        self.assertEqual(blocked_task.status, 'idle')

    def test_to_json(self):
        self.kanban.add_task("Test Task")
        json_str = self.kanban.to_json()
        self.assertIn("Test Task", json_str)

    def test_from_json(self):
        self.kanban.add_task("Test Task")
        json_str = self.kanban.to_json()
        new_kanban = WeaveKanban(self.agent)
        new_kanban.from_json(json_str)
        self.assertEqual(len(new_kanban.tasks), 1)
        self.assertEqual(new_kanban.tasks[0].title, "Test Task")

if __name__ == '__main__':
    unittest.main()
