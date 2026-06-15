"""Tests du serveur MCP — allègement du retour (_compact)."""

from score7_mcp import server


def test_compact_trims_bulky_fields():
    """_compact retire les champs verbeux et tronque la grille, sans perte sur disque."""
    r = {
        "chords": [{"chord": "C", "chord_full": "C", "beats": 1} for _ in range(50)],
        "melody": {"notes": [{"pitch": 60}], "sequence": "C4 E4 G4", "n_notes": 1,
                   "pitch_classes": {"C": 1}},
        "rhythm_pattern": {"bands": {"kick": {"pattern": [0.0] * 16, "grid": "X...",
                                              "density_hz": 1.0}}},
    }
    out = server._compact(r)
    assert len(out["chords"]) == server._CHORDS_RETURN_CAP
    assert out["chords_total"] == 50
    assert "notes" not in out["melody"] and "sequence" not in out["melody"]
    assert out["melody"]["pitch_classes"] == {"C": 1}  # résumé conservé
    assert "pattern" not in out["rhythm_pattern"]["bands"]["kick"]
    assert out["rhythm_pattern"]["bands"]["kick"]["grid"] == "X..."  # grille lisible gardée


def test_compact_leaves_short_grids_intact():
    """Une grille courte n'est pas tronquée et ne reçoit pas de champ chords_total."""
    r = {"chords": [{"chord": "Am", "beats": 4}]}
    out = server._compact(r)
    assert len(out["chords"]) == 1
    assert "chords_total" not in out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
