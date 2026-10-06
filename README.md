# nexus-smi-omega
Générer sources (Python/C++/Rust/JAX/CUDA)  [2] Build binaires (C++/Rust/CUDA)  [3] Collecter infos système  [4] Afficher tableau SMI  [5] Analyse JAX  [6] Générer 10 configurations  [7] Lancer les 4 tests  [8] Pipeline complet (1→7)  [9] Voir data JSON  [10] Voir configs  [11] Voir logs  [12] Quitter


    NEXUS SMI OMEGA est un outil d'inspection système haute performance,
    conçu comme alternative complète à nvidia-smi — sans dépendance propriétaire.
    Il agrège CPU, GPU multi-vendor, mémoire, stockage et métriques thermiques
    via un pipeline multi-langage unifié.


Démarrage rapide · Documentation · Architecture · Configurations · Tests
Présentation

Les outils de monitoring GPU sont historiquement liés à l'écosystème NVIDIA. NEXUS SMI OMEGA brise cette dépendance en s'appuyant exclusivement sur les interfaces kernel Linux (/proc, /sys, /sys/class/drm) et en exposant un pipeline multi-langage extensible.

Chaque couche du pipeline est indépendante et interopérable :

                          ┌─────────────────────────────────┐
                          │       nexus_smi_omega.sh        │
                          │     Orchestrateur principal     │
                          └─────────────┬───────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
    ┌──────▼───────┐          ┌─────────▼────────┐       ┌──────────▼───────┐
    │  Python Core │          │   C++17 Engine   │       │    Rust Probe    │
    │  smi_core.py │          │ smi_engine.cpp   │       │  smi_probe.rs    │
    │              │          │                  │       │                  │
    │  /proc · /sys│          │  Multi-thread    │       │  Memory-safe     │
    │  JSON export │          │  JSON export     │       │  JSON export     │
    └──────┬───────┘          └─────────┬────────┘       └──────────┬───────┘
           │                            │                            │
           └────────────────────────────▼────────────────────────────┘
                                        │
                          ┌─────────────▼───────────────┐
                          │       JAX Analyzer          │
                          │     jax_analyzer.py         │
                          │                             │
                          │  Health Score · Entropy     │
                          │  L2 Norm · Backend auto     │
                          └─────────────────────────────┘
                                        │
                          ┌─────────────▼───────────────┐
                          │       CUDA Probe (opt.)     │
                          │       cuda_probe.cu         │
                          │  NVIDIA VRAM · Compute Cap  │
                          └─────────────────────────────┘

Fonctionnalités
Domaine 	Métriques collectées
CPU 	Modèle, cœurs, fréquence (current/min/max), governor, load average 1/5/15m, usage % instantané
GPU 	Détection multi-vendor via /sys/class/drm, VRAM, température, fallback lspci
Mémoire 	Total, disponible, usage %, données brutes /proc/meminfo
Stockage 	Liste /sys/block, taille, modèle, rotation, df -h par point de montage
Thermique 	Toutes thermal_zone* (type + température), tous hwmon*
Analyse 	Norme L2, entropie Shannon, health score 0–100, backend JAX ou numpy
Configs 	10 profils JSON prêts à l'emploi (realtime, benchmark, CI/CD, ML…)
Tests 	Suite de 4 tests intégrés : syntaxe, collecte, binaires, analyse
GPU — Vendors supportés
Vendor ID 	Fabricant détecté
0x10de 	NVIDIA
0x1002 	AMD / ATI
0x8086 	Intel
0x1af4 	Virtio
Architecture
Structure des fichiers

nexus-smi-omega/
│
├── nexus_smi_omega.sh          ← Point d'entrée unique
│
└── nexus_smi/                  ← Généré au premier lancement
    │
    ├── src/                    ← Sources générées
    │   ├── smi_core.py         Python  — collecte système complète
    │   ├── jax_analyzer.py     Python  — analyse matricielle (JAX/numpy)
    │   ├── smi_engine.cpp      C++17   — moteur multi-thread
    │   ├── smi_probe.rs        Rust    — sonde memory-safe
    │   └── cuda_probe.cu       CUDA    — probe GPU NVIDIA (optionnel)
    │
    ├── build/                  ← Binaires compilés
    │   ├── smi_cpp
    │   ├── smi_rust
    │   └── smi_cuda
    │
    ├── data/                   ← Sorties JSON
    │   ├── smi.json            Données Python
    │   ├── smi_cpp.json        Données C++
    │   ├── smi_rust.json       Données Rust
    │   └── smi_cuda.txt        Données CUDA
    │
    ├── configs/                ← 10 profils de configuration
    │   ├── 01_realtime.json
    │   ├── 02_deep_scan.json
    │   └── ...
    │
    ├── tests/                  ← Artefacts de test
    ├── cache/
    └── logs/
        └── smi.log

Modèle de données

Toutes les sorties respectent un schéma JSON commun :

{
  "timestamp": "2026-10-06T10:30:00Z",
  "host": "hostname",
  "system": { "os": "Linux", "release": "6.6.0", "machine": "x86_64", "uptime": "..." },
  "cpu": {
    "info": { "model": "Intel Core i9-13900K", "cores": 24 },
    "freq": { "current": "4800000", "min": "800000", "max": "5800000", "governor": "performance" },
    "usage_percent": 18.4,
    "loadavg": "1.23 0.98 0.81 3/1024 42198"
  },
  "gpu": [
    { "name": "card0", "vendor": "AMD/ATI", "vendor_id": "0x1002", "vram_total_gb": 16.0, "temperature_c": 52.0 }
  ],
  "memory": { "total_gb": 64.0, "available_gb": 48.2, "usage_percent": 24.7 },
  "disks": [ { "name": "nvme0n1", "size_gb": 953.87, "model": "Samsung 980 Pro", "rotational": "0" } ],
  "disk_usage": [ { "filesystem": "/dev/nvme0n1p2", "size": "932G", "used": "210G", "avail": "675G", "use_pct": "24%", "mount": "/" } ],
  "thermal": {
    "zones": [ { "name": "thermal_zone0", "type": "acpitz", "temp_c": 29.8 } ],
    "hwmon": [ { "name": "coretemp", "temps": [31.0, 33.0, 29.0] } ]
  }
}

Démarrage rapide
Prérequis

Obligatoires

    bash ≥ 4.0
    python3 ≥ 3.7

Optionnels — détectés automatiquement, le script fonctionne sans eux

# Compilateur C++17
apt install g++

# Compilateur Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Analyse matricielle
pip install numpy              # Fallback léger
pip install jax jaxlib         # Accélération complète

# Détection GPU avancée
apt install pciutils           # fournit lspci

Installation

git clone https://github.com/DGK-Aissa/nexus-smi-omega.git
cd nexus-smi-omega
chmod +x nexus_smi_omega.sh

Premier lancement

# Pipeline complet — recommandé
./nexus_smi_omega.sh --full

Ce pipeline exécute en séquence :

    Génération des 5 sources
    Compilation C++ / Rust / CUDA
    Collecte système → JSON
    Affichage tableau SMI
    Analyse JAX
    Génération des 10 configurations
    Suite de 4 tests

Documentation
Interface ligne de commande

Usage : ./nexus_smi_omega.sh [OPTION]

OPTIONS :
  --gen         Générer les 5 fichiers source
  --build       Compiler les binaires (C++ / Rust / CUDA)
  --collect     Collecter les métriques système
  --show        Collecter et afficher le tableau SMI
  --jax         Lancer l'analyse JAX
  --configs     Générer les 10 configurations JSON
  --tests       Lancer la suite de 4 tests
  --full        Exécuter le pipeline complet (--gen → --tests)
  --version     Afficher la version
  --help        Afficher cette aide

Sans option : ouvre le menu interactif.

Menu interactif

./nexus_smi_omega.sh

╔══════════════════════════════════════════════════════════════════════╗
║  ⚡ NEXUS SMI OMEGA — v2.0.0                                         ║
║  Sans NVIDIA · 10 configs · 4 tests · Multi-langage                 ║
╚══════════════════════════════════════════════════════════════════════╝

  [1]  Générer sources (Python/C++/Rust/JAX/CUDA)
  [2]  Build binaires (C++/Rust/CUDA)
  [3]  Collecter infos système
  [4]  Afficher tableau SMI
  [5]  Analyse JAX
  [6]  Générer 10 configurations
  [7]  Lancer les 4 tests
  [8]  Pipeline complet (1→7)
  [9]  Voir data JSON
  [10] Voir configs
  [11] Voir logs
  [12] Quitter

Composants sources
smi_core.py — Collecte Python

Module de collecte principal. Lit directement /proc et /sys sans dépendance externe.

python3 nexus_smi/src/smi_core.py          # Résumé texte
python3 nexus_smi/src/smi_core.py --json   # Sortie JSON complète

Méthode 	Source noyau 	Description
collect_cpu() 	/proc/cpuinfo, /proc/stat, /sys/.../cpufreq 	Modèle, cœurs, usage, fréquence
collect_gpu() 	/sys/class/drm, lspci 	Détection multi-vendor, VRAM, température
collect_memory() 	/proc/meminfo 	Total, disponible, usage %
collect_disks() 	/sys/block, df -h 	Liste, taille, modèle, montages
collect_thermal() 	/sys/class/thermal, /sys/class/hwmon 	Toutes les zones thermiques
jax_analyzer.py — Analyse matricielle

Analyse les métriques système sous forme de vecteur d'état. Utilise JAX si disponible, sinon numpy en fallback transparent.

python3 nexus_smi/src/jax_analyzer.py

{
  "state_norm": 72.483,
  "entropy": 1.041,
  "balanced": true,
  "health_score": 68.4,
  "backend": "numpy 1.26.4"
}

Métriques calculées :
Métrique 	Formule 	Interprétation
state_norm 	‖[cpu, mem, gpu]‖₂ 	Charge globale du système
entropy 	-Σ p·log(p+ε) 	Équilibre entre les ressources
balanced 	cpu < 70% ∧ mem < 80% 	Système stable
health_score 	100 - (cpu×0.4 + mem×0.4 + gpu×0.2) 	Score santé 0–100

Grille d'interprétation du health score :
Score 	État 	Action recommandée
80 – 100 	🟢 Excellent 	Aucune
60 – 79 	🟡 Bon 	Surveillance légère
40 – 59 	🟠 Dégradé 	Identifier la ressource sous pression
0 – 39 	🔴 Critique 	Intervention requise
smi_engine.cpp — Moteur C++17

Lecture multi-thread des mêmes sources avec std::mutex. Performances supérieures sur systèmes à haute charge.

g++ -O3 -std=c++17 -pthread nexus_smi/src/smi_engine.cpp -o nexus_smi/build/smi_cpp
./nexus_smi/build/smi_cpp
./nexus_smi/build/smi_cpp --json

smi_probe.rs — Sonde Rust

Mêmes métriques avec garanties de sûreté mémoire du compilateur Rust.

rustc -O nexus_smi/src/smi_probe.rs -o nexus_smi/build/smi_rust
./nexus_smi/build/smi_rust
./nexus_smi/build/smi_rust --json

cuda_probe.cu — Probe CUDA (optionnel)

Récupère les propriétés des GPU NVIDIA via le CUDA Runtime API.

nvcc -O3 nexus_smi/src/cuda_probe.cu -o nexus_smi/build/smi_cuda
./nexus_smi/build/smi_cuda

CUDA devices: 2
  [0] NVIDIA RTX 4090
      Compute: 8.9
      VRAM: 24.00 GB
  [1] NVIDIA RTX 3080
      Compute: 8.6
      VRAM: 10.00 GB

Configurations

10 profils JSON générés dans nexus_smi/configs/, couvrant tous les cas d'usage courants.

./nexus_smi_omega.sh --configs

# 	Profil 	Intervalle 	Sources 	Export
01 	Realtime Monitoring 	1 s 	cpu · memory · gpu · thermal 	json
02 	Deep Scan 	60 s 	tous (cpu · mem · gpu · disks · network · thermal · pci · usb) 	json + csv
03 	Performance Profile 	5 s 	cpu · memory · gpu 	json — avec analyse JAX
04 	Thermal Watch 	2 s 	thermal · power 	json — alerte à 80 °C
05 	GPU Focus 	1 s 	gpu — CUDA activé 	json
06 	CI/CD Integration 	snapshot 	cpu · memory · disks 	json — exit si CPU > 90 %
07 	Benchmark 	300 s 	cpu · memory · gpu · disks 	json + csv — stats avg/min/max/p95/p99
08 	Alerting 	10 s 	cpu · memory · thermal 	json — seuils CPU 85 % / RAM 90 % / temp 80 °C
09 	Archive 	3 600 s 	tous 	json — rotation 30 j + compression
10 	ML Analysis 	60 s 	cpu · memory · gpu · disks 	json — JAX, détection d'anomalies, fenêtre 100 pts
Exemple — 06_ci_cd.json

{
  "name": "CI/CD Integration",
  "interval_sec": 0,
  "collect": ["cpu", "memory", "disks"],
  "export": "json",
  "exit_on_high": true,
  "threshold_cpu_pct": 90
}

Utilisation dans un pipeline GitHub Actions :

- name: Vérification ressources
  run: |
    ./nexus_smi_omega.sh --collect
    ./nexus_smi_omega.sh --jax

Tests

./nexus_smi_omega.sh --tests

Test 1 — Syntaxe Python

Valide la syntaxe de smi_core.py et jax_analyzer.py via python3 -m py_compile.

[🧪] ✅ smi_core.py : OK
[🧪] ✅ jax_analyzer.py : OK

Test 2 — Collecte et validation JSON

Exécute smi_core.py --json, vérifie la taille du fichier produit et parse le JSON.

[🧪] ✅ Collecte OK : 4 821 octets
[🧪] ✅ JSON valide

Test 3 — Binaires C++ et Rust

Vérifie l'existence des exécutables et leur exécution sans erreur.

[🧪] ✅ smi_cpp existe
[🧪] ✅ smi_rust existe
[🧪] ✅ smi_cpp exécute
[🧪] ✅ smi_rust exécute

Test 4 — Analyse JAX

Exécute jax_analyzer.py et vérifie le code de retour.

[🧪] ✅ JAX analyzer fonctionne

Résultat

═══ RÉSULTAT TESTS ═══
4/4 tests passés

Exemples de sortie
Tableau SMI

+-----------------------------------------------------------------------------+
| NEXUS-SMI  my-workstation                                                   |
+-----------------------------------------------------------------------------+
| CPU: Intel Core i9-13900K                              | Cores: 24   |
| Usage: 18.4   % | Load: 1.23 0.98 0.81 3/1024 42198              |
| RAM: 64.00  GB  | Free: 48.20  GB  | Usage: 24.7 %                 |
| GPUs: 1
|   [0] AMD card0 — VRAM: 16.0 GB
| Disks: 2
|   nvme0n1       953.87     GB  (Samsung 980 Pro)
|   sda            931.51    GB  (Seagate Barracuda)
| Temps: thermal_zone0=29.8°C  thermal_zone1=31.0°C
+-----------------------------------------------------------------------------+

Logs structurés

[2026-10-06T10:30:00Z] [OK]   smi.json (4821 octets)
[2026-10-06T10:30:00Z] [OK]   smi_cpp (C++)
[2026-10-06T10:30:01Z] [OK]   smi_rust (Rust)
[2026-10-06T10:30:01Z] [INFO] CUDA absent (normal si pas de GPU NVIDIA)
[2026-10-06T10:30:02Z] [OK]   10 configurations générées
[2026-10-06T10:30:03Z] [OK]   4/4 tests passés

Dépannage

smi.json absent lors de l'analyse JAX

# Toujours lancer la collecte avant l'analyse
./nexus_smi_omega.sh --collect && ./nexus_smi_omega.sh --jax

Erreur de compilation C++ (-lstdc++fs manquant)

# Sur GCC < 9, le filesystem expérimental requiert un flag supplémentaire
g++ -O3 -std=c++17 -pthread nexus_smi/src/smi_engine.cpp -lstdc++fs -o nexus_smi/build/smi_cpp

JAX et numpy absents

pip install numpy          # Solution minimale
pip install jax jaxlib     # Solution complète

Aucun GPU détecté
Comportement normal sur machines sans GPU dédié ou en VM.
Le script poursuit l'exécution sans interruption.

Rust : rustc introuvable

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

Contribution

Les contributions sont les bienvenues. Merci de respecter ces conventions :

# 1. Fork + clone
git clone https://github.com/DGK-Aissa/nexus-smi-omega.git

# 2. Créer une branche
git checkout -b feat/nom-de-la-feature

# 3. Développer + tester
./nexus_smi_omega.sh --tests

# 4. Commit (convention Conventional Commits)
git commit -m "feat(jax): ajouter calcul de variance"

# 5. Ouvrir une Pull Request

Types de commits acceptés : feat · fix · docs · refactor · test · chore
Licence

Distribué sous licence MIT. Voir LICENSE pour le texte complet.

Conçu et développé par Aissa Mohammedi (DGK) — 2026

NEXUS SMI OMEGA — Parce que votre infrastructure mérite un outil à sa hauteur.
