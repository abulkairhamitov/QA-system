# QA-system
QA system based on sberquad database.

To load the docker image, just execute the following commands specifying the env file with TELEGRAM_TOKEN:
```bash
  docker pull abulkair/my_telegram_bot
  docker run --env-file .env -it my_telegram_bot
```
You can test the telegram bot by clicking here https://t.me/QAsystem_sberquad_bot (waiting time of 5 minutes due to 2 gb RAM)
