"""
Mapping des types d'attaques bruts (BigFlow-NIDS-V2 ET CICIDS2017/
CICFlowMeter) vers les 5 catégories officielles du projet. Module partagé
entre tous les pipelines (nettoyage, entraînement, adaptateurs) : une seule
source de vérité, pour éviter toute divergence entre le mapping utilisé pour
les visualisations et celui utilisé pour l'entraînement réel.

Déplacé depuis random_forest/scripts/mapping_attaques.py (qui redirige ici
via un shim) lors du passage au socle de features canonique multi-source :
ce module n'est plus spécifique à Random Forest, il sert aussi à construire
la vérité terrain catégorielle du jeu d'entraînement combiné.
"""

TYPES_ATTAQUES = [
    'brute_force',
    'port_scanning',
    'suspicious_activity',
    'intrusion_reseau',
    'acces_non_autorise',
]

MAPPING = {
    'ftp-bruteforce': 'brute_force', 'ssh-bruteforce': 'brute_force',
    'password': 'brute_force', 'brute_force_-web': 'brute_force',
    'brute_force_-xss': 'brute_force',
    'scanning': 'port_scanning', 'reconnaissance': 'port_scanning',
    'bot': 'suspicious_activity', 'generic': 'suspicious_activity',
    'analysis': 'suspicious_activity', 'mitm': 'suspicious_activity',
    'fuzzers': 'suspicious_activity',
    'ddos': 'intrusion_reseau', 'dos': 'intrusion_reseau',
    'ddos_attack-hoic': 'intrusion_reseau', 'ddos_attacks-loic-http': 'intrusion_reseau',
    'dos_attacks-slowhttptest': 'intrusion_reseau', 'dos_attacks-hulk': 'intrusion_reseau',
    'dos_attacks-goldeneye': 'intrusion_reseau', 'dos_attacks-slowloris': 'intrusion_reseau',
    'ddos_attack-loic-udp': 'intrusion_reseau', 'exploits': 'intrusion_reseau',
    'injection': 'intrusion_reseau', 'xss': 'intrusion_reseau',
    'sql_injection': 'intrusion_reseau', 'shellcode': 'intrusion_reseau',
    'worms': 'intrusion_reseau',
    'backdoor': 'acces_non_autorise', 'infilteration': 'acces_non_autorise',
    'theft': 'acces_non_autorise', 'ransomware': 'acces_non_autorise',
    # Libellés du dataset CICIDS2017 (format CICFlowMeter, différent de
    # BigFlow-NIDS-V2) : mêmes 5 catégories, noms d'attaques différents.
    'ftp-patator': 'brute_force', 'ssh-patator': 'brute_force',
    'infiltration': 'acces_non_autorise', 'heartbleed': 'intrusion_reseau',
    'portscan': 'port_scanning',
    # Les libellés "Web Attack - Brute Force/XSS/Sql Injection" ne sont pas
    # mappés en dur ici : le tiret cadratin d'origine se corrompt de façon
    # non déterministe selon l'encodage de lecture du CSV, donc une
    # correspondance exacte serait fragile. La règle de repli ci-dessous
    # (mot-clé 'brute'/'xss'/'injection') les catégorise correctement quel
    # que soit le caractère de séparation effectivement lu.
}

# Règles de repli (mot-clé -> catégorie) si un type brut n'est pas dans
# MAPPING (ex : nouvelle variante d'attaque non encore répertoriée).
_FALLBACK = [
    ('brute', 'brute_force'), ('password', 'brute_force'),
    ('patator', 'brute_force'),
    ('scan', 'port_scanning'), ('recon', 'port_scanning'),
    ('dos', 'intrusion_reseau'), ('ddos', 'intrusion_reseau'),
    ('injection', 'intrusion_reseau'), ('exploit', 'intrusion_reseau'),
    ('shell', 'intrusion_reseau'), ('worm', 'intrusion_reseau'),
    ('xss', 'intrusion_reseau'),
    ('bot', 'suspicious_activity'), ('fuzz', 'suspicious_activity'),
    ('backdoor', 'acces_non_autorise'), ('infiltrat', 'acces_non_autorise'),
    ('theft', 'acces_non_autorise'), ('ransom', 'acces_non_autorise'),
]


def mapper(nom):
    nom = str(nom).strip().lower()
    if nom in MAPPING:
        return MAPPING[nom]
    for mot, cat in _FALLBACK:
        if mot in nom:
            return cat
    return 'suspicious_activity'
