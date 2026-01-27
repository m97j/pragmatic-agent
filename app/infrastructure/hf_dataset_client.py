# app/infrastructure/hf_dataset_client.py
import json
import os

from huggingface_hub import HfApi

from app.config import FALLBACK_DATASET_ID as fb_ds
from app.config import HF_TOKEN
from app.config import SESSIONS_DATASET_ID as ds


class SessionManager:
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.api = HfApi()
        self.session_list = []

    def slugify(self, text: str) -> str:
        """Convert text to a slug suitable for filenames."""
        if text == "Untitled":
            return ""
        text = text.replace(" ", "__SPC__")
        return text
    
    def _unslugify(self, slug: str) -> str:
        """Convert slug back to a readable title."""
        return slug.replace("__SPC__", " ").title()

    def list_sessions(self, hf_user):
        """List all sessions for a user by reading filenames only."""
        try:
            files = self.api.list_repo_files(
                repo_id=f"{hf_user}/{ds}",
                repo_type="dataset",
                token=self.hf_token,
            )
            # File naming convention: hf_user-title-session_id-timestamp.json
            sessions = []
            for f in files:
                filename = os.path.basename(f)
                if filename.endswith(".json") and filename.startswith(f"{hf_user}-"):
                    parts = filename.replace(".json", "").split("-")
                    if len(parts) >= 4:
                        _, title_slug, session_id, timestamp = parts
                        title = self._unslugify(title_slug)
                        sessions.append((title, session_id, timestamp))
            
            # Sort by timestamp descending
            sessions.sort(key=lambda x: x[2], reverse=True)

            # Dropdown Choices only return the form (title, session_id)
            self.session_list = [(title, session_id) for title, session_id, _ in sessions]
            return self.session_list
        except Exception:
            try:
                files = self.api.list_repo_files(
                    repo_id=fb_ds,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                # File naming convention: hf_user-title-session_id-timestamp.json
                sessions = []
                for f in files:
                    filename = os.path.basename(f)
                    if filename.endswith(".json") and filename.startswith(f"{hf_user}-"):
                        parts = filename.replace(".json", "").split("-")
                        if len(parts) >= 4:
                            _, title_slug, session_id, timestamp = parts
                            title = self._unslugify(title_slug)
                            sessions.append((title, session_id, timestamp))
                
                # Sort by timestamp descending
                sessions.sort(key=lambda x: x[2], reverse=True)

                # Dropdown Choices only return the form (title, session_id)
                self.session_list = [(title, session_id) for title, session_id, _ in sessions]
                return self.session_list
            except Exception:
                return self.session_list
        
    def add_session(self, title, session_id):
        """Add a new session to the list."""
        self.session_list.append((title, session_id))

    def get_sessions(self):
        """Get the current list of sessions."""
        return self.session_list

    def download_session(self, hf_user, session_id):
        """Download a specific session file and return records."""
        try:
            files = self.api.list_repo_files(
                repo_id=f"{hf_user}/{ds}",
                repo_type="dataset",
                token=self.hf_token,
            )
            # Find the latest file for the session_id
            target_files = [f for f in files if f.startswith(f"{hf_user}-") and f.endswith(f"-{session_id}.json") and f.split("-")[-2] == session_id]
            if not target_files:
                return []
            
            target_files.sort(key=lambda f: f.split("-")[-1].replace(".json", ""), reverse=True)
            latest_file = target_files[0]

            local_path = self.api.hf_hub_download(
                repo_id=f"{hf_user}/{ds}",
                repo_type="dataset",
                filename=f"sessions/{latest_file}",
                token=self.hf_token
            )
            with open(local_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            return records
        except Exception:
            try:
                files = self.api.list_repo_files(
                    repo_id=fb_ds,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                # Find the latest file for the session_id
                target_files = [f for f in files if f.startswith(f"{hf_user}-") and f.endswith(f"-{session_id}.json") and f.split("-")[-2] == session_id]
                if not target_files:
                    return []
                
                target_files.sort(key=lambda f: f.split("-")[-1].replace(".json", ""), reverse=True)
                latest_file = target_files[0]

                local_path = self.api.hf_hub_download(
                    repo_id=fb_ds,
                    repo_type="dataset",
                    filename=f"sessions/{latest_file}",
                    token=HF_TOKEN
                )
                with open(local_path, "r", encoding="utf-8") as f:
                    records = json.load(f)

                return records
            except Exception:
                return []

    def push_session(self, hf_user, session_id, records, title_raw, timestamp, backup=False):
        """Push session history as a separate file to Hub."""
        try:
            ds_id = f"{hf_user}/{ds}" if not backup else fb_ds
            token = self.hf_token if not backup else HF_TOKEN
            title_slug = self.slugify(title_raw or "untitled")
            filename = f"sessions/{hf_user}-{title_slug}-{session_id}-{timestamp}.json"
            tmp_path = f"/tmp/{filename}"

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            self.api.create_repo(
                repo_id=ds_id,
                repo_type="dataset",
                token=token,
                exist_ok=True
            )

            self.api.upload_file(
                path_or_fileobj=tmp_path,
                repo_id=ds_id,
                repo_type="dataset",
                path_in_repo=filename,
                token=token
            )
            os.remove(tmp_path)

            files = self.api.list_repo_files(
                repo_id=ds_id,
                repo_type="dataset",
                token=token,
            )
            prefix = f"{hf_user}-{title_slug}-{session_id}-"
            old_files = [f for f in files if f.startswith(prefix) and f != filename]
            for old_file in old_files:
                try:
                    self.api.delete_file(
                        repo_id=ds_id,
                        repo_type="dataset",
                        path_in_repo=f"sessions/{old_file}",
                        token=token
                    )
                except Exception:
                    pass
        except Exception:
            pass
