# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for AlphaFold.predict_multiseed and make_empty_prev."""

from absl.testing import absltest
from absl.testing import parameterized
from alphafold.common import residue_constants
from alphafold.model import config as model_config
from alphafold.model import modules
from alphafold.model import modules_multimer
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _get_small_config(num_recycle=0):
  """Returns a tiny model config for fast unit tests."""
  cfg = model_config.model_config('model_1')
  c = cfg.model
  # Minimise computation.
  c.embeddings_and_evoformer.evoformer_num_block = 1
  c.embeddings_and_evoformer.extra_msa_stack_num_block = 1
  c.embeddings_and_evoformer.template.enabled = False
  c.num_recycle = num_recycle
  # Disable heads that require ground-truth labels.
  c.heads.masked_msa.weight = 0.0
  c.heads.distogram.weight = 0.0
  c.heads.predicted_lddt.weight = 0.0
  c.heads.predicted_aligned_error.weight = 0.0
  c.heads.experimentally_resolved.weight = 0.0
  return cfg.model


def _make_batch(num_residues, num_msa, num_extra_msa, num_ensemble=1):
  """Creates a minimal synthetic batch for the AlphaFold monomer model.

  All arrays have a leading ensemble dimension because AlphaFoldIteration
  uses ``slice_batch(i)`` which indexes into axis 0 of every entry in
  ``ensembled_batch``.
  """
  emb_c = _get_small_config().embeddings_and_evoformer
  target_feat_dim = residue_constants.atom_type_num  # 37 features
  msa_feat_dim = 49  # standard MSA feature dimension

  def _tile(arr):
    """Add a leading ensemble dimension by tiling."""
    return np.tile(arr[None], [num_ensemble] + [1] * arr.ndim)

  batch = {
      # Ensembled (leading dimension = num_ensemble).
      'aatype': _tile(np.zeros([num_residues], dtype=np.int32)),
      'residue_index': _tile(np.arange(num_residues, dtype=np.int32)),
      'seq_length': np.full([num_ensemble], num_residues, dtype=np.int32),
      'seq_mask': _tile(np.ones([num_residues], dtype=np.float32)),
      'msa_mask': _tile(np.ones([num_msa, num_residues], dtype=np.float32)),
      'msa_feat': _tile(
          np.zeros([num_msa, num_residues, msa_feat_dim], dtype=np.float32)
      ),
      'target_feat': _tile(
          np.zeros([num_residues, target_feat_dim], dtype=np.float32)
      ),
      'extra_msa': _tile(
          np.zeros([num_extra_msa, num_residues], dtype=np.int32)
      ),
      'extra_msa_mask': _tile(
          np.ones([num_extra_msa, num_residues], dtype=np.float32)
      ),
      'extra_has_deletion': _tile(
          np.zeros([num_extra_msa, num_residues], dtype=np.float32)
      ),
      'extra_deletion_value': _tile(
          np.zeros([num_extra_msa, num_residues], dtype=np.float32)
      ),
      # Atom existence masks needed by StructureModule.
      'atom14_atom_exists': _tile(
          np.ones([num_residues, 14], dtype=np.float32)
      ),
      'atom37_atom_exists': _tile(
          np.ones([num_residues, 37], dtype=np.float32)
      ),
      # Atom-index mapping tables needed by all_atom.atom14_to_atom37 etc.
      'residx_atom37_to_atom14': _tile(
          np.zeros([num_residues, 37], dtype=np.int32)
      ),
      'residx_atom14_to_atom37': _tile(
          np.zeros([num_residues, 14], dtype=np.int32)
      ),
      # Recycling inputs: not ensembled; overridden by non_ensembled_batch
      # (i.e. batched_prev) in predict_multiseed, but still needed here as
      # placeholders so that slice_batch can handle them.
      'prev_pos': np.zeros(
          [num_residues, residue_constants.atom_type_num, 3], dtype=np.float32
      ),
      'prev_msa_first_row': np.zeros(
          [num_residues, emb_c.msa_channel], dtype=np.float32
      ),
      'prev_pair': np.zeros(
          [num_residues, num_residues, emb_c.pair_channel], dtype=np.float32
      ),
  }
  return batch


class MakeEmptyPrevTest(parameterized.TestCase):
  """Tests for the make_empty_prev helper."""

  @parameterized.named_parameters(
      ('single_seed', 1),
      ('multi_seed', 4),
  )
  def test_shapes(self, num_seeds):
    cfg = _get_small_config()
    emb_config = cfg.embeddings_and_evoformer
    num_residues = 10

    batched_prev = modules.make_empty_prev(emb_config, num_residues, num_seeds)

    if emb_config.recycle_pos:
      self.assertIn('prev_pos', batched_prev)
      self.assertEqual(
          batched_prev['prev_pos'].shape,
          (num_seeds, num_residues, residue_constants.atom_type_num, 3),
      )

    if emb_config.recycle_features:
      self.assertIn('prev_msa_first_row', batched_prev)
      self.assertEqual(
          batched_prev['prev_msa_first_row'].shape,
          (num_seeds, num_residues, emb_config.msa_channel),
      )
      self.assertIn('prev_pair', batched_prev)
      self.assertEqual(
          batched_prev['prev_pair'].shape,
          (num_seeds, num_residues, num_residues, emb_config.pair_channel),
      )

  def test_all_zeros(self):
    cfg = _get_small_config()
    emb_config = cfg.embeddings_and_evoformer
    batched_prev = modules.make_empty_prev(emb_config, num_residues=5, num_seeds=2)
    for v in batched_prev.values():
      np.testing.assert_array_equal(v, 0.0)


class PredictMultiseedTest(parameterized.TestCase):
  """Tests for AlphaFold.predict_multiseed."""

  def _run_predict_multiseed(self, num_seeds, shard_size, num_recycle=0):
    num_residues = 5
    num_msa = 3
    num_extra_msa = 4
    cfg = _get_small_config(num_recycle=num_recycle)
    emb_config = cfg.embeddings_and_evoformer

    batch = _make_batch(num_residues, num_msa, num_extra_msa)
    batched_prev = modules.make_empty_prev(emb_config, num_residues, num_seeds)

    def forward(batch, batched_prev):
      model = modules.AlphaFold(cfg)
      return model.predict_multiseed(
          batch,
          is_training=False,
          batched_prev=batched_prev,
          shard_size=shard_size,
      )

    init, apply = hk.transform(forward).init, hk.transform(forward).apply
    rng = jax.random.PRNGKey(0)
    params = init(rng, batch, batched_prev)
    result = apply(params, rng, batch, batched_prev)
    return result

  @parameterized.named_parameters(
      ('seeds_1_shard_1', 1, 1),
      ('seeds_2_shard_1', 2, 1),
      ('seeds_2_shard_2', 2, 2),
      ('seeds_3_shard_1', 3, 1),
  )
  def test_output_has_leading_seed_dim(self, num_seeds, shard_size):
    result = self._run_predict_multiseed(num_seeds, shard_size)
    # The structure_module output should have a leading N_seeds dimension.
    final_atom_positions = result['structure_module']['final_atom_positions']
    self.assertEqual(final_atom_positions.shape[0], num_seeds)

  def test_shard_size_does_not_affect_output(self):
    """Results for shard_size=1 and shard_size=N_seeds should be identical."""
    num_seeds = 2
    result_shard1 = self._run_predict_multiseed(num_seeds, shard_size=1)
    result_shardN = self._run_predict_multiseed(num_seeds, shard_size=num_seeds)

    positions_shard1 = result_shard1['structure_module']['final_atom_positions']
    positions_shardN = result_shardN['structure_module']['final_atom_positions']

    # Same parameters and inputs → same outputs regardless of shard_size.
    np.testing.assert_allclose(
        np.array(positions_shard1), np.array(positions_shardN), atol=1e-5
    )

  def test_output_shape_structure(self):
    """Each seed should produce a complete set of model outputs."""
    num_seeds = 2
    num_residues = 5
    result = self._run_predict_multiseed(num_seeds, shard_size=1)

    sm = result['structure_module']
    # final_atom_positions: [N_seeds, N_res, 37, 3]
    self.assertEqual(
        sm['final_atom_positions'].shape, (num_seeds, num_residues, 37, 3)
    )
    # final_atom_mask: [N_seeds, N_res, 37]
    self.assertEqual(
        sm['final_atom_mask'].shape, (num_seeds, num_residues, 37)
    )

  def test_num_recycle_no_error(self):
    """predict_multiseed should work when num_recycle > 0."""
    result = self._run_predict_multiseed(num_seeds=2, shard_size=1, num_recycle=1)
    self.assertIn('structure_module', result)
    self.assertEqual(result['structure_module']['final_atom_positions'].shape[0], 2)


# ---------------------------------------------------------------------------
# AlphaFold-Multimer tests
# ---------------------------------------------------------------------------


def _get_small_multimer_config(num_recycle=0):
  """Returns a tiny AlphaFold-Multimer config for fast unit tests."""
  cfg = model_config.model_config('model_1_multimer_v3')
  c = cfg.model
  # Minimise computation.
  c.embeddings_and_evoformer.evoformer_num_block = 1
  c.embeddings_and_evoformer.extra_msa_stack_num_block = 1
  c.embeddings_and_evoformer.template.enabled = False
  # Use very small MSA sizes so the test is fast.
  c.embeddings_and_evoformer.num_msa = 4
  c.embeddings_and_evoformer.num_extra_msa = 4
  c.num_recycle = num_recycle
  c.heads.masked_msa.weight = 0.0
  c.heads.distogram.weight = 0.0
  c.heads.predicted_lddt.weight = 0.0
  c.heads.predicted_aligned_error.weight = 0.0
  c.heads.experimentally_resolved.weight = 0.0
  return cfg.model


def _make_multimer_batch(num_residues, num_msa):
  """Creates a minimal synthetic batch for the AlphaFold-Multimer model.

  The multimer model takes a flat batch dict (no leading ensemble dim). MSA
  sampling is performed internally in JAX, so the batch must contain raw MSA
  arrays before sampling.

  Includes dummy ``all_atom_positions`` so that the structure module output
  is available to populate ``prev_pos`` during recycling.
  """
  cfg = _get_small_multimer_config()
  emb_c = cfg.embeddings_and_evoformer
  atom_type_num = residue_constants.atom_type_num  # 37

  batch = {
      # Sequence features.
      'aatype': np.zeros([num_residues], dtype=np.int32),
      'residue_index': np.arange(num_residues, dtype=np.int32),
      'seq_mask': np.ones([num_residues], dtype=np.float32),
      # MSA features: shape [N_msa, N_res] before internal sampling.
      'msa': np.zeros([num_msa, num_residues], dtype=np.int32),
      'msa_mask': np.ones([num_msa, num_residues], dtype=np.float32),
      'deletion_matrix': np.zeros([num_msa, num_residues], dtype=np.float32),
      # Extra MSA (= full MSA before sampling; sample_msa splits it).
      'extra_msa': np.zeros([num_msa, num_residues], dtype=np.int32),
      'extra_msa_mask': np.ones([num_msa, num_residues], dtype=np.float32),
      'extra_deletion_matrix': np.zeros(
          [num_msa, num_residues], dtype=np.float32
      ),
      # Chain / entity identity features used by relative position encoding.
      'asym_id': np.zeros([num_residues], dtype=np.float32),
      'entity_id': np.zeros([num_residues], dtype=np.float32),
      'sym_id': np.zeros([num_residues], dtype=np.float32),
      # Dummy ground-truth atoms – their presence as a key allows the
      # structure module output to be used for representation updates.
      'all_atom_positions': np.zeros(
          [num_residues, atom_type_num, 3], dtype=np.float32
      ),
      'all_atom_mask': np.zeros([num_residues, atom_type_num], dtype=np.float32),
      # Recycling placeholders overridden by batched_prev in predict_multiseed.
      'prev_pos': np.zeros(
          [num_residues, atom_type_num, 3], dtype=np.float32
      ),
      'prev_msa_first_row': np.zeros(
          [num_residues, emb_c.msa_channel], dtype=np.float32
      ),
      'prev_pair': np.zeros(
          [num_residues, num_residues, emb_c.pair_channel], dtype=np.float32
      ),
  }
  return batch


class MultiseedMultimerTest(parameterized.TestCase):
  """Tests for modules_multimer.AlphaFold.predict_multiseed."""

  def _run_predict_multiseed(self, num_seeds, shard_size, num_recycle=0):
    num_residues = 5
    num_msa = 8
    cfg = _get_small_multimer_config(num_recycle=num_recycle)
    emb_config = cfg.embeddings_and_evoformer

    batch = _make_multimer_batch(num_residues, num_msa)
    batched_prev = modules.make_empty_prev(emb_config, num_residues, num_seeds)

    def forward(batch, batched_prev):
      model = modules_multimer.AlphaFold(cfg)
      return model.predict_multiseed(
          batch,
          is_training=False,
          batched_prev=batched_prev,
          shard_size=shard_size,
      )

    init, apply = hk.transform(forward).init, hk.transform(forward).apply
    rng = jax.random.PRNGKey(0)
    params = init(rng, batch, batched_prev)
    result = apply(params, rng, batch, batched_prev)
    return result

  @parameterized.named_parameters(
      ('seeds_1_shard_1', 1, 1),
      ('seeds_2_shard_1', 2, 1),
      ('seeds_2_shard_2', 2, 2),
  )
  def test_output_has_leading_seed_dim(self, num_seeds, shard_size):
    result = self._run_predict_multiseed(num_seeds, shard_size)
    final_atom_positions = result['structure_module']['final_atom_positions']
    self.assertEqual(final_atom_positions.shape[0], num_seeds)

  def test_output_shape_structure(self):
    """Each seed should produce a complete set of model outputs."""
    num_seeds = 2
    num_residues = 5
    result = self._run_predict_multiseed(num_seeds, shard_size=1)

    sm = result['structure_module']
    # final_atom_positions: [N_seeds, N_res, 37, 3]
    self.assertEqual(
        sm['final_atom_positions'].shape, (num_seeds, num_residues, 37, 3)
    )
    # final_atom_mask: [N_seeds, N_res, 37]
    self.assertEqual(
        sm['final_atom_mask'].shape, (num_seeds, num_residues, 37)
    )

  def test_shard_size_does_not_affect_output(self):
    """Results should be identical regardless of shard_size."""
    num_seeds = 2
    result_shard1 = self._run_predict_multiseed(num_seeds, shard_size=1)
    result_shardN = self._run_predict_multiseed(num_seeds, shard_size=num_seeds)

    positions_shard1 = result_shard1['structure_module']['final_atom_positions']
    positions_shardN = result_shardN['structure_module']['final_atom_positions']

    np.testing.assert_allclose(
        np.array(positions_shard1), np.array(positions_shardN), atol=1e-5
    )

  def test_num_recycle_no_error(self):
    """predict_multiseed should work when num_recycle > 0."""
    result = self._run_predict_multiseed(num_seeds=2, shard_size=1, num_recycle=1)
    self.assertIn('structure_module', result)
    self.assertEqual(
        result['structure_module']['final_atom_positions'].shape[0], 2
    )


if __name__ == '__main__':
  absltest.main()
