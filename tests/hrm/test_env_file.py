from nla.hrm.openai_compat import _key_from_env_file


def test_env_file_plain_quoted_and_export_lines(tmp_path):
    f = tmp_path / ".env"
    f.write_text('OTHER=1\nexport FAKE_KEY="abc123"\n')
    assert _key_from_env_file("FAKE_KEY", str(f)) == "abc123"
    f.write_text("FAKE_KEY='xyz'\n")
    assert _key_from_env_file("FAKE_KEY", str(f)) == "xyz"
    f.write_text("FAKE_KEY_2=nope\n")
    assert _key_from_env_file("FAKE_KEY", str(f)) is None      # a longer name must not match
    assert _key_from_env_file("FAKE_KEY", str(tmp_path / "missing")) is None
