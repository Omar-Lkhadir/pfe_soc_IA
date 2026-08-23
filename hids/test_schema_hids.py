"""
Test de l'agrégation HIDS sur des événements synthétiques (construits à la
main d'après les vrais noms de champs Filebeat system module, cf.
canonical_schema_hids.py) -- vérifie que le passage événement-par-événement
-> features-par-fenêtre produit des valeurs cohérentes, avant tout
branchement sur de vraies données étiquetées.

Exécution : python hids/test_schema_hids.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hids.canonical_schema_hids import agreger_fenetre_ssh, agreger_fenetre_sudo

erreurs = []


def check(nom, cond, detail=''):
    status = 'OK' if cond else 'ECHEC'
    print(f'  [{status}] {nom} {detail}')
    if not cond:
        erreurs.append(f'{nom} {detail}')


# ============================================================
print('=== SSH : scénario brute-force (20 échecs puis 1 réussite, 1 IP, 10s) ===')
base = pd.Timestamp('2026-08-23 10:00:00')
evenements = []
for i in range(20):
    evenements.append({
        'timestamp': base + pd.Timedelta(seconds=i * 0.5),
        'source_ip': '203.0.113.5', 'user_name': f'admin{i % 3}',
        'ssh_event': 'Failed', 'ssh_method': 'password',
    })
evenements.append({
    'timestamp': base + pd.Timedelta(seconds=10.5),
    'source_ip': '203.0.113.5', 'user_name': 'admin0',
    'ssh_event': 'Accepted', 'ssh_method': 'password',
})
df_brute = pd.DataFrame(evenements)
out_brute = agreger_fenetre_ssh(df_brute, fenetre='5min')

check('1 seule fenêtre produite (toutes les 21 lignes dans la même minute)', len(out_brute) == 1)
r = out_brute.iloc[0]
check('nb_tentatives == 21', r['nb_tentatives'] == 21, f"(obtenu {r['nb_tentatives']})")
check('nb_echecs == 20', r['nb_echecs'] == 20, f"(obtenu {r['nb_echecs']})")
check('reussite_apres_echecs == True (signal clé du brute-force)', r['reussite_apres_echecs'] == True)
check('ratio_echec > 0.9', r['ratio_echec'] > 0.9, f"(obtenu {r['ratio_echec']:.2f})")
check('intervalle_moyen_sec < 1s (rythme automatisé)', r['intervalle_moyen_sec'] < 1.0,
      f"(obtenu {r['intervalle_moyen_sec']:.2f}s)")
check('nb_utilisateurs_distincts == 3', r['nb_utilisateurs_distincts'] == 3)

# ============================================================
print('\n=== SSH : scénario normal (1 échec de frappe, 1 réussite, 1 IP, 40s) ===')
evenements_normal = [
    {'timestamp': base, 'source_ip': '198.51.100.9', 'user_name': 'alice',
     'ssh_event': 'Failed', 'ssh_method': 'password'},
    {'timestamp': base + pd.Timedelta(seconds=40), 'source_ip': '198.51.100.9', 'user_name': 'alice',
     'ssh_event': 'Accepted', 'ssh_method': 'password'},
]
df_normal = pd.DataFrame(evenements_normal)
out_normal = agreger_fenetre_ssh(df_normal, fenetre='5min')
r2 = out_normal.iloc[0]
check('nb_tentatives == 2 (normal)', r2['nb_tentatives'] == 2)
check('intervalle_moyen_sec == 40s (rythme humain, PAS automatisé)', abs(r2['intervalle_moyen_sec'] - 40.0) < 1e-6,
      f"(obtenu {r2['intervalle_moyen_sec']:.1f}s)")
check('reussite_apres_echecs == True aussi (1 typo, pas un signal fort seul -- '
      "d'où l'intérêt de nb_tentatives/intervalle en plus, pas cette feature isolée)",
      r2['reussite_apres_echecs'] == True)

print('\n   -> Différenciateur clé entre brute-force et faute de frappe : PAS')
print('      "reussite_apres_echecs" seule (vraie dans les deux cas), mais')
print('      nb_tentatives (21 vs 2) et intervalle_moyen_sec (0.5s vs 40s).')

# ============================================================
print('\n=== SUDO : scénario abus (accès root répété, commandes variées) ===')
evenements_sudo = [
    {'timestamp': base + pd.Timedelta(minutes=i), 'user_name': 'bob',
     'sudo_user': 'root', 'sudo_command': f'/usr/bin/cmd{i}', 'sudo_error': None}
    for i in range(8)
]
df_sudo = pd.DataFrame(evenements_sudo)
out_sudo = agreger_fenetre_sudo(df_sudo, fenetre='1h')
rs = out_sudo.iloc[0]
check('nb_commandes_sudo == 8', rs['nb_commandes_sudo'] == 8)
check('ratio_cible_root == 1.0', rs['ratio_cible_root'] == 1.0)
check('nb_commandes_distinctes == 8', rs['nb_commandes_distinctes'] == 8)
check('nb_echecs_sudo == 0', rs['nb_echecs_sudo'] == 0)

# ============================================================
print(f"\n{'='*60}")
if erreurs:
    print(f"ECHEC : {len(erreurs)} assertion(s) en erreur :")
    for e in erreurs:
        print(f'  - {e}')
    sys.exit(1)
else:
    print("TOUS LES TESTS PASSENT")
