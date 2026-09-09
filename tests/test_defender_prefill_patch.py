"""The frozen defender must not be prefilled with the CoT scaffold."""

import importlib.util
import pathlib
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "patch_defender_prefill",
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "slurm" / "patch_defender_prefill.py",
)
patch_defender_prefill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patch_defender_prefill)

UPSTREAM = '''def convert_game_history_to_messages(history, player_role, prompt, prompt_type, custom_configs=None, tokenizer=None):
    messages = convert_game_history_to_query(history, player_role, prompt, prompt_type, custom_configs)

    chat_message = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    if custom_configs and custom_configs.get("direct_chat_no_cot", False):
        pass
    else:
        chat_message += ASSISTANT_THINKING_PREFIX

    return chat_message
'''


def _render(source, player_role, custom_configs):
    """Execute the patched function body against stub collaborators."""
    namespace = {
        "convert_game_history_to_query": lambda *a, **k: [],
        "ASSISTANT_THINKING_PREFIX": "Let's think step by step.\n<think> ",
        "tokenizer": None,
    }
    stub_tokenizer = type(
        "T", (), {"apply_chat_template": staticmethod(lambda *a, **k: "RENDERED")}
    )()
    exec(source, namespace)
    return namespace["convert_game_history_to_messages"](
        [], player_role, "seed", "generated_harmful", custom_configs, stub_tokenizer
    )


def _configs(trained_role):
    return {
        "optimizer_train_role": trained_role,
        "no_defender_turn": trained_role == "attacker",
        "no_attacker_turn": False,
        "direct_chat_no_cot": False,
    }


class DefenderPrefillPatchTest(unittest.TestCase):
    def setUp(self):
        self.patched, changed = patch_defender_prefill.apply(UPSTREAM)
        self.assertTrue(changed)

    def test_upstream_prefills_every_role(self):
        # The behaviour being corrected: upstream prefills the frozen defender.
        rendered = _render(UPSTREAM, "defender", _configs("attacker"))
        self.assertTrue(rendered.endswith("<think> "))

    def test_frozen_defender_is_not_prefilled(self):
        rendered = _render(self.patched, "defender", _configs("attacker"))
        self.assertEqual(rendered, "RENDERED")

    def test_trainable_defender_keeps_its_prefill(self):
        rendered = _render(self.patched, "defender", _configs("defender"))
        self.assertTrue(rendered.endswith("<think> "))

    def test_attacker_keeps_its_prefill_in_both_phases(self):
        # _render_attacker_prompt prefills in the payoff path, so training must
        # match it whether the attacker is the trainable role or the opponent.
        for trained in ("attacker", "defender"):
            with self.subTest(trained=trained):
                rendered = _render(self.patched, "attacker", _configs(trained))
                self.assertTrue(rendered.endswith("<think> "))

    def test_direct_chat_no_cot_still_suppresses_everything(self):
        for role in ("attacker", "defender"):
            configs = dict(_configs("attacker"), direct_chat_no_cot=True)
            with self.subTest(role=role):
                self.assertEqual(_render(self.patched, role, configs), "RENDERED")

    def test_patch_is_idempotent(self):
        again, changed = patch_defender_prefill.apply(self.patched)
        self.assertFalse(changed)
        self.assertEqual(again, self.patched)

    def test_moved_upstream_is_an_error_not_a_silent_noop(self):
        with self.assertRaises(SystemExit):
            patch_defender_prefill.apply("def f():\n    return 1\n")


if __name__ == "__main__":
    unittest.main()
