import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
FP8_PATH = REPO_ROOT / "python/sglang/srt/layers/quantization/fp8.py"
MEGA_MOE_PATH = REPO_ROOT / "python/sglang/srt/layers/moe/mega_moe.py"


class TestMegaMoeInvariants(unittest.TestCase):
    def _parse(self, path: pathlib.Path):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return source, tree

    def test_sm90_all_fp8_trigger_condition_is_guarded(self):
        """Ensure non-FP4 mega build path is SM90-only and excludes SM100."""
        source, tree = self._parse(FP8_PATH)

        condition_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_src = ast.get_source_segment(source, node.test) or ""
                if (
                    "not self.is_fp4_expert" in test_src
                    and "envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()" in test_src
                    and "is_sm90_supported()" in test_src
                    and "not is_sm100_supported()" in test_src
                ):
                    condition_found = True
                    break

        self.assertTrue(
            condition_found,
            "Expected non-FP4 MegaMoE trigger to be guarded by SM90 and not SM100.",
        )

    def test_k_dimension_rule_differs_for_fp8_vs_fp4(self):
        """Ensure K derivation uses K=last_dim for SM90 FP8, K=last_dim*2 otherwise."""
        source, tree = self._parse(MEGA_MOE_PATH)

        def collect_assign_exprs(nodes):
            result = {}
            for n in nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1:
                    target = n.targets[0]
                    if isinstance(target, ast.Name):
                        result[target.id] = ast.get_source_segment(source, n.value)
            return result

        target_if = None
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_src = ast.get_source_segment(source, node.test) or ""
                if test_src.strip() != "use_sm90_fp8_mega":
                    continue
                body_assigns = collect_assign_exprs(node.body)
                if body_assigns.get("k1") == "half_k1":
                    target_if = node
                    break

        self.assertIsNotNone(
            target_if,
            "Missing `if use_sm90_fp8_mega` branch for K dimension derivation.",
        )

        body_assigns = collect_assign_exprs(target_if.body)
        orelse_assigns = collect_assign_exprs(target_if.orelse)

        self.assertEqual(
            body_assigns.get("k1"),
            "half_k1",
            "SM90 FP8 path must set k1 = half_k1.",
        )
        self.assertEqual(
            body_assigns.get("k2"),
            "half_k2",
            "SM90 FP8 path must set k2 = half_k2.",
        )
        self.assertEqual(
            orelse_assigns.get("k1"),
            "half_k1 * 2",
            "Non-SM90-FP8 path must set k1 = half_k1 * 2.",
        )
        self.assertEqual(
            orelse_assigns.get("k2"),
            "half_k2 * 2",
            "Non-SM90-FP8 path must set k2 = half_k2 * 2.",
        )

    def test_sm90_memory_share_branch_exists(self):
        """The SM90 fp8 memory-share branch must (1) interleave the L1 fp8
        weight (gate/up gran-8 along N), (2) write the interleaved fp8 weight
        back into ``experts.w13_weight.data`` and point ``mega_l1_weights[0]``
        at the same buffer (weight memory share), (3) physically share the
        block-(128, 128) SF with deep-ep by pointing ``mega_l{1,2}_weights[1]``
        at ``experts.w{13,2}_weight_scale_inv.data`` without copying or
        reassigning those tensors, and (4) NOT invoke
        ``_transpose_sf_for_utccp`` (SM90 has no UTCCP path).
        """
        source = MEGA_MOE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get() and use_sm90_fp8_mega",
            source,
            "Missing SM90 memory-share branch in build_mega_moe_experts_weights.",
        )

        # Find the SM90 memory-share branch body and validate its key statements.
        sm90_branch_marker = "elif envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get() and use_sm90_fp8_mega:"
        idx = source.index(sm90_branch_marker)
        # Take a window that should contain only this branch.
        branch_src = source[idx : idx + 2000]

        self.assertIn(
            "_interleave_l1_weights((w13, w13_sf))",
            branch_src,
            "SM90 memory-share branch must call _interleave_l1_weights on the "
            "(weight, per-row SF) pair.",
        )
        self.assertIn(
            "experts.w13_weight.data = w13_interleaved",
            branch_src,
            "SM90 memory-share branch must write back interleaved L1 weight.",
        )
        self.assertIn(
            "experts.mega_l1_weights = (",
            branch_src,
            "SM90 memory-share branch must (re)assign mega_l1_weights.",
        )
        # Strip comment lines so docstring/explanatory mentions of UTCCP/SF
        # sharing do not trip the "no call" / "no SF reassignment" checks.
        non_comment_src = "\n".join(
            line
            for line in branch_src.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "_transpose_sf_for_utccp(",
            non_comment_src,
            "SM90 has no UTCCP; the memory-share branch must not call "
            "_transpose_sf_for_utccp.",
        )
        # SF MUST be shared with deep-ep on SM90: mega_l{1,2}_weights[1] must
        # point at the original `experts.w{13,2}_weight_scale_inv.data`
        # tensors. The branch must not reassign those buffers (sharing is by
        # aliasing the existing .data, not by replacing it).
        self.assertIn(
            "experts.w13_weight_scale_inv.data,",
            non_comment_src,
            "SM90 memory-share branch must alias "
            "experts.w13_weight_scale_inv.data into mega_l1_weights for SF sharing.",
        )
        self.assertIn(
            "experts.w2_weight_scale_inv.data,",
            non_comment_src,
            "SM90 memory-share branch must alias "
            "experts.w2_weight_scale_inv.data into mega_l2_weights for SF sharing.",
        )
        self.assertNotIn(
            "experts.w13_weight_scale_inv.data =",
            non_comment_src,
            "SM90 memory-share branch must not reassign "
            "experts.w13_weight_scale_inv.data (sharing is via aliasing .data).",
        )
        self.assertNotIn(
            "experts.w2_weight_scale_inv.data =",
            non_comment_src,
            "SM90 memory-share branch must not reassign "
            "experts.w2_weight_scale_inv.data (sharing is via aliasing .data).",
        )


if __name__ == "__main__":
    unittest.main()
