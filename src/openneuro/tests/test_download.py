"""Test downloading and authentication."""

import json
from pathlib import Path
from unittest import mock

import pytest

import openneuro
import openneuro._config
from openneuro import download
from openneuro._download import _skip_directory

dataset_id_aws = "ds000246"
tag_aws = "1.0.0"
include_aws = "sub-0001/anat"
exclude_aws = []

dataset_id_on = "ds000117"
tag_on = None
include_on = "sub-16/ses-meg"
exclude_on = "*.fif"  # save GBs of downloads

invalid_tag = "abcdefg"


@pytest.mark.parametrize(
    ("dataset_id", "tag", "include", "exclude"),
    [
        (dataset_id_aws, tag_aws, include_aws, exclude_aws),
        (dataset_id_on, tag_on, include_on, exclude_on),
    ],
)
def test_download(tmp_path: Path, dataset_id, tag, include, exclude):
    """Test downloading some files."""
    download(
        dataset=dataset_id,
        tag=tag,
        target_dir=tmp_path,
        include=include,
        exclude=exclude,
    )


def test_download_invalid_tag(
    tmp_path: Path, dataset_id=dataset_id_aws, invalid_tag=invalid_tag
):
    """Test handling of a non-existent tag."""
    with pytest.raises(RuntimeError, match="snapshot.*does not exist"):
        download(dataset=dataset_id, tag=invalid_tag, target_dir=tmp_path)


def test_resume_download(tmp_path: Path):
    """Test resuming of a dataset download."""
    dataset = "ds000246"
    tag = "1.0.0"
    include = ["CHANGES"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)

    # Download some more files
    include = ["sub-0001/meg/*.jpg"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)

    # Download from a different revision / tag
    new_tag = "00001"
    include = ["CHANGES"]
    with pytest.raises(FileExistsError, match=f"revision {tag} exists"):
        download(dataset=dataset, tag=new_tag, target_dir=tmp_path, include=include)

    # Try to "resume" from a different dataset
    new_dataset = "ds000117"
    with pytest.raises(RuntimeError, match="existing dataset.*appears to be different"):
        download(dataset=new_dataset, target_dir=tmp_path, include=include)

    # Remove "DatasetDOI" from JSON
    json_path = tmp_path / "dataset_description.json"
    with json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    del dataset_json["DatasetDOI"]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f)

    with pytest.raises(RuntimeError, match=r'does not contain "DatasetDOI"'):
        download(dataset=dataset, target_dir=tmp_path)

    # We should be able to resume a download even if "datset_description.jon"
    # is missing
    json_path.unlink()
    include = ["sub-0001/meg/sub-0001_coordsystem.json"]
    download(dataset=dataset, tag=tag, target_dir=tmp_path, include=include)


def test_ds000248(tmp_path: Path):
    """Test a dataset for that we ship default excludes."""
    dataset = "ds000248"
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)


def test_doi_handling(tmp_path: Path):
    """Test that we can handle DOIs that start with 'doi:`."""
    dataset = "ds000248"
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)

    # Now inject a `doi:` prefix into the DOI
    dataset_description_path = tmp_path / "dataset_description.json"
    dataset_description_text = dataset_description_path.read_text(encoding="utf-8")
    dataset_description = json.loads(dataset_description_text)
    # Make sure we can dumps to get the same thing back (if they change their
    # indent 4->8 for example, we might try to resume our download of the file
    # and things will break in a challenging way)
    dataset_description_rt = json.dumps(dataset_description, indent=4)
    assert dataset_description_text == dataset_description_rt
    # Ensure the dataset doesn't already have the problematic prefix, then add
    assert not dataset_description["DatasetDOI"].startswith("doi:")
    dataset_description["DatasetDOI"] = "doi:" + dataset_description["DatasetDOI"]
    dataset_description_path.write_text(
        data=json.dumps(dataset_description, indent=4), encoding="utf-8"
    )

    # Try to download again
    download(dataset=dataset, include=["participants.tsv"], target_dir=tmp_path)


def test_restricted_dataset(tmp_path: Path, openneuro_token: str):
    """Test downloading a restricted dataset."""
    with mock.patch.object(openneuro._config, "CONFIG_PATH", tmp_path / ".openneuro"):
        with mock.patch("getpass.getpass", lambda _: openneuro_token):
            openneuro._config.init_config()

        # This is a restricted dataset that is only available if the API token
        # was used correctly.
        download(dataset="ds006412", include="README.txt", target_dir=tmp_path)

    assert (tmp_path / "README.txt").exists()


@pytest.mark.parametrize(
    ("root", "include_patterns", "dir_path", "expected"),
    [
        # Test Case 1: Root-level directory traversal
        # When root is empty, we're at the top level
        ("", ["sub-01"], "sub-01", False),  # sub-01 matches include, so don't skip
        ("", ["sub-01"], "sub-02", True),   # sub-02 doesn't match include, so skip
        ("", ["sub-01/ses-meg"], "sub-01/ses-meg", False),  # matches, don't skip
        ("", ["sub-01/ses-meg"], "sub-01/ses-mri", True),   # doesn't match, skip
        
        # Test Case 2: Directory is a Parent of the Include Pattern
        ("", ["sub-01/*"], "sub-01", False),  # sub-01 is parent of sub-01/*, don't skip
        ("", ["sub-01/ses-meg/*"], "sub-01", False),  # sub-01 is parent, don't skip
        ("", ["sub-01/ses-meg/*"], "sub-01/ses-mri", True),  # doesn't match pattern, skip
        ("", ["sub-01/ses-meg/*"], "sub-01/ses-meg", False),  # matches, don't skip
        ("", ["sub-01/ses-meg/*"], "sub-01/ses-meg/meg", False),  # matches, don't skip
        ("", ["sub-01/*"], "sub-02", True),  # doesn't match, skip
        ("", ["sub-01/ses-mri/*"], "sub-01/ses-meg", True),  # doesn't match, skip
        
        # Test Case 3: Directory or Subdirectory Match (No Wildcards)
        ("", ["sub-01/ses-emg"], "sub-01", False),  # sub-01 is parent of sub-01/ses-emg, don't skip
        ("", ["sub-01/ses-emg/"], "sub-01", False),  # same with trailing slash
        ("", ["sub-01/ses-meg"], "sub-01/ses-mri", True),  # doesn't match, skip
        ("", ["sub-01/ses-emg"], "sub-01/ses-emg", False),  # exact match, don't skip
        ("", ["sub-01/ses-emg/"], "sub-01/ses-emg", False),  # exact match with slash, don't skip
        ("sub-01", ["sub-01/ses-emg"], "sub-01/ses-emg/meg", False),  # sub-01/ses-emg is parent, don't skip
        ("sub-01", ["sub-01/ses-emg/"], "sub-01/ses-emg/meg", False),  # same with slash
        ("", ["sub-01/ses-emg"], "sub-02/ses-emg", True),  # doesn't match, skip
        
        # Test Case 4: Wildcard Pattern Prefix Match
        ("", ["sub-01/*"], "sub-01/ses-meg", False),  # matches sub-01/*, don't skip
        ("sub-01", ["sub-01/*"], "sub-01/ses-meg/meg", False),  # matches sub-01/*, don't skip
        ("", ["sub-01/*"], "sub-02/ses-meg", True),  # doesn't match, skip
        ("", ["sub-01/ses-*"], "sub-01/ses-meg", False),  # matches sub-01/ses-*, don't skip
        ("", ["sub-01/ses-*"], "sub-01/ses-mri", False),  # matches sub-01/ses-*, don't skip
        ("", ["sub-01/ses-*"], "sub-01/anat", True),  # doesn't match, skip
        ("sub-01", ["sub-01/ses-*"], "sub-01/ses-meg/meg", False),  # matches, don't skip
        ("sub-01", ["sub-01/ses-meg/*"], "sub-01/ses-meg/meg", False),  # matches, don't skip
        ("sub-01", ["sub-01/ses-mri/*"], "sub-01/ses-meg/meg", True),  # doesn't match, skip
        
        # Test Case 5: Nested directory traversal (root evolves with dir_path)
        # When we're inside sub-01, root becomes "sub-01"
        ("sub-01", ["sub-01/ses-meg"], "sub-01/ses-meg", False),  # exact match, don't skip
        ("sub-01", ["sub-01/ses-meg"], "sub-01/ses-mri", True),   # doesn't match, skip
        ("sub-01", ["sub-01/ses-*"], "sub-01/ses-meg", False),   # matches sub-01/ses-*, don't skip
        ("sub-01", ["sub-01/ses-*"], "sub-01/ses-mri", False),   # matches sub-01/ses-*, don't skip
        ("sub-01", ["sub-01/ses-*"], "sub-01/anat", True),       # doesn't match, skip
        
        # When we're inside sub-01/ses-meg, root becomes "sub-01/ses-meg"
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg"], "sub-01/ses-meg/meg", False),  # exact match, don't skip
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg"], "sub-01/ses-meg/anat", True),  # doesn't match, skip
        ("sub-01/ses-meg", ["sub-01/ses-meg/*"], "sub-01/ses-meg/meg", False),    # matches, don't skip
        ("sub-01/ses-meg", ["sub-01/ses-meg/*"], "sub-01/ses-meg/anat", False),   # matches, don't skip
        
        # Test Case 6: Deep nesting with evolving root
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg/raw"], "sub-01/ses-meg/meg/raw", False),  # exact match
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg/raw"], "sub-01/ses-meg/meg/processed", True),  # doesn't match
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg/*"], "sub-01/ses-meg/meg/raw", False),     # matches wildcard
        ("sub-01/ses-meg", ["sub-01/ses-meg/meg/*"], "sub-01/ses-meg/anat", True),         # doesn't match
        
        # Edge Cases
        ("", [""], "", False),  # Empty paths match, don't skip
        ("", ["sub-01"], "", False),  # Empty dir_path matches any include, don't skip
        ("", ["sub-01/"], "sub-01", False),  # Trailing slash matches, don't skip
        ("", ["sub-01"], "sub-01/", False),  # Trailing slash on dir_path matches, don't skip
        ("", ["sub-01/"], "sub-01/", False),  # Both with trailing slash match, don't skip
        
        # Deep nesting tests with empty root
        ("", ["sub-01/*"], "sub-01/ses-meg/meg/raw", False),  # matches sub-01/*, don't skip
        ("", ["sub-01/ses-*"], "sub-01/ses-meg/meg/raw", False),  # matches sub-01/ses-*, don't skip
        ("", ["sub-01/ses-meg/*"], "sub-01/ses-meg/meg/raw", False),  # matches, don't skip
        ("", ["sub-01/ses-meg/meg/*"], "sub-01/ses-meg/meg/raw", False),  # matches, don't skip
        ("", ["sub-01/ses-mri/*"], "sub-01/ses-meg/meg/raw", True),  # doesn't match, skip
        ("", ["sub-02/*"], "sub-01/ses-meg/meg/raw", True),  # doesn't match, skip
        
        # Complex wildcard patterns
        ("", ["sub-*"], "sub-01/ses-meg", False),  # matches sub-*, don't skip
        ("", ["sub-01/ses-*"], "sub-01/ses-meg", False),  # matches sub-01/ses-*, don't skip
        
        # Special characters and edge cases
        ("", ["sub-01_special"], "sub-01_special", False),  # exact match, don't skip
        ("", ["sub-01-special"], "sub-01-special", False),  # exact match, don't skip
        ("", ["sub-01.special"], "sub-01.special", False),  # exact match, don't skip
        ("", ["sub-01-special"], "sub-01_special", True),  # Different separators, skip
        ("", ["sub-01_special"], "sub-01-special", True),  # Different separators, skip
        
        # Multiple wildcards (should match prefix before first *)
        ("", ["sub-01/*/*"], "sub-01/ses-meg/meg", False),  # matches sub-01/*/*, don't skip
        ("", ["sub-01/ses-*/*"], "sub-01/ses-meg/meg", False),  # matches sub-01/ses-*/*, don't skip
        
        # Very deep paths
        ("", ["a/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/*, don't skip
        ("", ["a/b/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/*, don't skip
        ("", ["a/b/c/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/*, don't skip
        ("", ["a/b/c/d/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/*, don't skip
        ("", ["a/b/c/d/e/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/*, don't skip
        ("", ["a/b/c/d/e/f/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/*, don't skip
        ("", ["a/b/c/d/e/f/g/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/g/*, don't skip
        ("", ["a/b/c/d/e/f/g/h/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/g/h/*, don't skip
        ("", ["a/b/c/d/e/f/g/h/i/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/g/h/i/*, don't skip
        ("", ["a/b/c/d/e/f/g/h/i/j/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/g/h/i/j/*, don't skip
        ("", ["a/b/c/d/e/f/g/h/i/j/k/*"], "a/b/c/d/e/f/g/h/i/j", False),  # matches a/b/c/d/e/f/g/h/i/j/k/*, don't skip
        ("", ["b/*"], "a/b/c/d/e/f/g/h/i/j", True),  # Wrong prefix, skip
        
        # Test with non-empty root (simulating deeper traversal)
        ("dataset", ["dataset/sub-01"], "dataset/sub-01", False),  # matches, don't skip
        ("dataset", ["dataset/sub-01"], "dataset/sub-02", True),   # doesn't match, skip
        ("dataset", ["dataset/sub-01/*"], "dataset/sub-01", False),  # matches, don't skip
        ("dataset", ["dataset/sub-01/*"], "dataset/sub-01/ses-meg", False),  # matches, don't skip
        ("dataset", ["dataset/sub-01/*"], "other/sub-01", True),   # doesn't match, skip
        
        # Test with mismatched root and dir_path (should handle gracefully)
        ("sub-01", ["sub-02/*"], "sub-01/ses-meg", True),  # root doesn't match include pattern, skip
        ("sub-01", ["sub-01/ses-meg"], "sub-02/ses-meg", True),  # dir_path doesn't match root, skip
    ],
)
def test_skip_directory(root: str, include_patterns: list[str], dir_path: str, expected: bool):
    """Test _skip_directory function with various directory paths and include patterns.
    
    This comprehensive test covers all the different cases handled by the function:
    1. Exact directory match
    2. Directory is a parent of the include pattern
    3. Directory or subdirectory match (no wildcards)
    4. Wildcard pattern prefix match
    5. Nested directory traversal with evolving root
    6. Deep nesting scenarios
    
    Note: The function returns True when a directory should be SKIPPED (not traversed),
    and False when it should be processed.
    
    The relationship between root and dir_path:
    - root represents the current directory being processed in the traversal
    - dir_path is the full path of the directory to check
    - As we traverse deeper, root evolves to become the previous dir_path
    - When root is empty (""), we're at the top level of the dataset
    
    Parameters
    ----------
    root : str
        The current directory being processed in the traversal (evolves as we go deeper)
    include_patterns : list[str]
        The include patterns to match against
    dir_path : str
        The full path of the directory to check
    expected : bool
        Expected result (True if directory should be skipped, False if it should be processed)
    """
    result = _skip_directory(root, include_patterns, dir_path)
    assert result == expected, (
        f"_skip_directory('{root}', {include_patterns}, '{dir_path}') "
        f"returned {result}, expected {expected} "
        f"(True=skip directory, False=process directory)"
    )
