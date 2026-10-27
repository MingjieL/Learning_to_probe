# Compact partition storage

File-backed router training now saves a `fixed-partition-v6` artifact containing:

```text
artifact.json
bucket_assignment.npy
```

The model/checkpoint and policy files are unchanged. The original base file is
an external dependency, referenced by a path relative to the partition directory.
Do not remove or change that base file while using the compact partition.
The base format stays unchanged: in particular, SpaceV remains int8 on disk.

On load, the partition rebuilds its CPU IVF using the stored bucket assignments
and original vectors. It does not retrain k-means or change assignments. Bucket
lengths, offsets, and posting IDs are reconstructed in memory. Original ID order
within each bucket is preserved, including its effect on tied-distance results.
No centroids are stored or reconstructed. Dimension comes from validated metadata;
the FAISS quantizer stays empty. Router-directed `search_preassigned` uses the
reconstructed posting lists, including adaptive policies, without centroid lookup. Vector conversion and
insertion use bounded batches; this avoids materializing an additional complete
float32 copy of the source, but the complete IVF still requires RAM.

This saves disk space at the cost of longer startup. The per-query search
implementation, routing model, and policy calculations remain unchanged.

## Existing artifacts and checkpoints

The loader accepts the existing self-contained v4, compact v5, and centroid-free v6.
All formats use the same algorithm compatibility profile and basename. A
storage conversion therefore does not require model retraining or policy refitting.
Pre-profile/otherwise unsupported formats remain unsupported.

Export an existing partition to a **different parent directory with the same
basename**:

```bash
python -m ft_anns compact-partition \
  --partition artifacts/deep100m/deep100m_bucket65536_niter10_pseed42_fixed-partition \
  --base data/deep100m/base.100M.fbin \
  --output artifacts/compact_v6/deep100m/deep100m_bucket65536_niter10_pseed42_fixed-partition
```

The command loads the source index, verifies **all** source vectors against its
stored vectors in bounded chunks, and writes the new artifact. It rejects a
source whose count/content differs and a posting order that assignments alone
cannot reproduce. This validation can be I/O intensive for 100M vectors. It
does not overwrite or delete the original partition or base data.

Point the existing train/search configuration's `partition_path` (or the CLI
`--partition` argument) at the new directory. Keep the same checkpoint and
policy. The original artifacts still occupy disk until separately retired;
adding a compact export alone does not reclaim their space.

## Python API

```python
partition.save(output_path, base_path=original_base_path)  # compact v6
replayed = FixedPartition.load(output_path)

# An explicit locator override can be used if the base moved:
replayed = FixedPartition.load(output_path, base_path=moved_base_path)

# Without an external base, save remains self-contained for in-memory callers:
partition.save(other_path)  # full v4 (only when the partition still has centroids)
```

Supported external formats: fbin, i8bin, ivecs, fvecs, bvecs. The loader validates
the base filename, shape/format and file length. As with the project's other
artifacts (ADR-0001), it does not hash content on every load: replacing a base
with different contents of the same name/shape/length is outside the compatibility
guarantee. Export verifies content against the source index, but the external
base must remain immutable afterward. Relative directory relationships must
be retained when moving the bundle, or the explicit Python locator override
must be supplied.

## Storage accounting

For v6, count the assignment, metadata, model/policy, and **one copy
of the external base file**. Do not add an estimated IVF or derived arrays that
are never written. A base shared among multiple artifacts is counted once when
measuring the whole deployment, but once per independent method when comparing
standalone deployments.

A loaded v6 partition exposes `centroids=None`. Re-export it with `base_path`
to v6; saving it as full v4 raises a clear error instead of fabricating centroids.
Existing v4/v5 artifacts retain their original centroids and remain readable.

Generated conversion reports, local deployment paths, and benchmark output
from the development workspace are not part of this source release.
