import os

import pytest

from incubator.control_panel.flexiv_compat import (
    FlexivElementsError,
    normalize_elements_version,
    select_flexiv_elements,
)


def test_elements_normalization_does_not_imply_an_rdk_version() -> None:
    assert normalize_elements_version("FlexivElementsStudio_v3.10.4") == (
        "3.10",
        "3.10.4",
    )


def test_explicit_elements_version_wins_without_host_discovery(tmp_path) -> None:
    selection = select_flexiv_elements(
        "3.11", environment={}, home=tmp_path, proc_root=tmp_path / "proc"
    )
    assert selection.elements_series == "3.11"
    assert selection.detected_version == "3.11"
    assert selection.source == "explicit"


def test_running_elements_process_wins_over_old_logs(tmp_path) -> None:
    home = tmp_path / "home"
    log = (
        home
        / "FSP_v3.9.3/FlexivElementsStudio_v3.9.3/"
        "FlexivElementsStudio/log/UIApp_old.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text("old", encoding="utf-8")
    command = tmp_path / "proc/123/cmdline"
    command.parent.mkdir(parents=True)
    command.write_bytes(
        b"/opt/FSP_v3.10.2/FlexivElementsStudio_v3.10.2/"
        b"FlexivElementsStudio\0"
    )

    selection = select_flexiv_elements(
        environment={}, home=home, proc_root=tmp_path / "proc"
    )

    assert selection.elements_series == "3.10"
    assert selection.source == "running-process"


def test_newest_elements_log_selects_the_profile(tmp_path) -> None:
    home = tmp_path / "home"
    older = (
        home
        / "FSP_v3.9.3/FlexivElementsStudio_v3.9.3/"
        "FlexivElementsStudio/log/UIApp_old.log"
    )
    newer = (
        home
        / "FSP_v3.11.1/FlexivElementsStudio_v3.11.1/"
        "FlexivElementsStudio/log/UIApp_new.log"
    )
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    older.touch()
    newer.touch()
    older_mtime = older.stat().st_mtime_ns
    newer_mtime = max(newer.stat().st_mtime_ns, older_mtime + 1_000_000_000)
    os.utime(newer, ns=(newer_mtime, newer_mtime))

    selection = select_flexiv_elements(
        environment={}, home=home, proc_root=tmp_path / "proc"
    )

    assert selection.elements_series == "3.11"
    assert selection.source == "elements-log"


def test_custom_elements_layout_reads_version_from_log_header(tmp_path) -> None:
    home = tmp_path / "home"
    log = home / "tools" / "FlexivElementsStudio" / "log" / "UIApp_latest.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        """[2026-07-12 15:15:08.720278][debug] Flexiv Robotics - Flexiv Elements
[2026-07-12 15:15:08.720280][debug] sw ver    : v3.10
[2026-07-12 15:15:08.720280][debug] git tag   : v3.10
""",
        encoding="utf-8",
    )

    selection = select_flexiv_elements(
        environment={}, home=home, proc_root=tmp_path / "proc"
    )

    assert selection.elements_series == "3.10"
    assert selection.detected_version == "3.10"
    assert selection.source == "elements-log"
    assert selection.source_path == str(log.resolve())


def test_multiple_installed_series_fail_closed_without_active_evidence(
    tmp_path,
) -> None:
    home = tmp_path / "home"
    (home / "FSP_v3.9.3").mkdir(parents=True)
    (home / "FSP_v3.10.2").mkdir(parents=True)

    with pytest.raises(FlexivElementsError, match="Multiple Flexiv"):
        select_flexiv_elements(
            environment={}, home=home, proc_root=tmp_path / "proc"
        )
