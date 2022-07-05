# Anomaly detection in high-energy physics using a quantum autoencoder
Quantum autoencoder (QAE) code used in the paper [arxiv: 2112.04958](https://arxiv.org/abs/2112.04958).
Background and one benchmark signal contained in 'data'.


To train and save default QAE implementation: 
>python auto_qml.py --train --save --train-size 1000 --save-dir ./check_save --trash-qubits 2 

To test saved run:
>python auto_qml.py --test --path ./check_save/MET_b1pt_lep1pt_lep2pt/train_1000/run_1/



If you use the code please cite:
>@article{Ngairangbam:2021yma,
    author = "Ngairangbam, Vishal S. and Spannowsky, Michael and Takeuchi, Michihisa",
    title = "{Anomaly detection in high-energy physics using a quantum autoencoder}",
    eprint = "2112.04958",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "OU-HET-1125, IPPP/21/54",
    doi = "10.1103/PhysRevD.105.095004",
    journal = "Phys. Rev. D",
    volume = "105",
    number = "9",
    pages = "095004",
    year = "2022"
}
