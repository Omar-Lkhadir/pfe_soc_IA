# Desactive la rotation des logs (conn.log reste le meme fichier en continu)
# -- Filebeat (filebeat-nids) tail un chemin fixe /var/log/zeek-live/conn.log,
# exactement comme pour la simulation zeek-replay ; une rotation horaire par
# defaut renommerait ce fichier et casserait ce chemin fixe.
redef Log::default_rotation_interval = 0secs;
