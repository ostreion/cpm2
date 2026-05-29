# Third-Party Dependencies

This directory contains vendored third-party code used by the CPNext pipeline.

## Included Libraries

### Protein-Hunter
- **Source**: https://github.com/yehlincho/Protein-Hunter
- **Version**: commit `xxxxxxx` (update with actual commit)
- **Date copied**: 2025-01-21
- **License**: MIT
- **Citation**:
  ```bibtex
  @article{cho2025protein,
    title={Protein Hunter: exploiting structure hallucination within diffusion for protein design},
    author={Cho, Yehlin and Rangel, Griffin and Bhardwaj, Gaurav and Ovchinnikov, Sergey},
    journal={bioRxiv},
    year={2025}
  }
  ```

### LigandMPNN (included in Protein-Hunter)
- **Source**: https://github.com/dauparas/LigandMPNN
- **License**: MIT
- **Citation**:
  ```bibtex
  @article{dauparas2023atomic,
    title={Atomic context-conditioned protein sequence design using LigandMPNN},
    author={Dauparas, Justas and others},
    journal={bioRxiv},
    year={2023}
  }
  ```

### cPEPmatch
- **Source**: https://github.com/briandasantini/cPEPmatch
- **Version**: commit `e3c746999afa0715d819c42f550d8c4ccae48489`
- **Date cloned**: 2026-01-21
- **License**: (not specified in repo)
- **Authors**: Brianda L. Santini, Prof. Dr. Martin Zacharias (TU Munich)
- **Citation**:
  ```bibtex
  @article{santini2020cpepmatch,
    title={Identification of the Binding Interface of MLL1-WDR5 Interaction Inhibitors: A Structure-Based Virtual Screening Approach},
    author={Santini, Brianda L and Zacharias, Martin},
    journal={Journal of Chemical Information and Modeling},
    year={2020},
    publisher={ACS Publications}
  }
  ```
- **Webserver**: https://t38webservices.nat.tum.de/cpepmatch/

### Boltz (boltz_ph - Protein-Hunter fork)
- **Source**: https://github.com/jwohlwend/boltz (original)
- **Modified by**: Protein-Hunter team
- **Version**: 2.2.1 (modified)
- **License**: MIT

## Updating Dependencies

To update a dependency:

1. Note the current version/commit in this file
2. Copy the new version into the appropriate subdirectory
3. Update the version info above
4. Test that the pipeline still works
5. Commit with message describing the update

## Installation Notes

Most dependencies require their own conda environment. See `../envs/` for environment specifications.

### Protein-Hunter setup
```bash
cd lib/Protein-Hunter
chmod +x setup.sh
./setup.sh
```

### LigandMPNN model weights
```bash
cd lib/Protein-Hunter/LigandMPNN
bash get_model_params.sh "./model_params"
```

### cPEPmatch setup
Requires Python 3.7 (note: older than other tools), Modeller license, and vmd-python.
```bash
conda create -n cpepmatch python=3.7.11
conda activate cpepmatch
cd lib/cPEPmatch
pip install -r requirements.txt
conda install -c conda-forge vmd-python
conda config --add channels salilab
conda install modeller
# Configure Modeller license key as instructed
```
