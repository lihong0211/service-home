#!/usr/bin/env python3
"""阿里云 DDNS - 仅更新 AAAA (IPv6)。从项目根目录 .env 读取 ALIYUN_ACCESS_KEY_ID、ALIYUN_ACCESS_KEY_SECRET。"""

import json
import logging
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkalidns.request.v20150109 import (
    DescribeDomainRecordsRequest,
    UpdateDomainRecordRequest,
    AddDomainRecordRequest,
)

CONFIG = {
    "REGION_ID": "cn-hangzhou",
    "DOMAIN": "home.doctor-dog.com",
    "SUBDOMAIN": "@",
    "TTL": 600,
    "CHECK_INTERVAL": 300,
    "IP_SERVICE": "https://api6.ipify.org",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class AliDNSClient:
    def __init__(self, access_key_id, access_key_secret, region_id="cn-hangzhou"):
        self.client = AcsClient(access_key_id, access_key_secret, region_id)

    def get_domain_records(self, domain_name, subdomain, record_type):
        request = DescribeDomainRecordsRequest.DescribeDomainRecordsRequest()
        request.set_DomainName(domain_name)
        request.set_accept_format("json")
        try:
            response = self.client.do_action_with_exception(request)
            data = json.loads(response.decode("utf-8"))
            rr_ok = (subdomain, "") if subdomain == "@" else (subdomain,)
            for r in data.get("DomainRecords", {}).get("Record", []):
                if r["RR"] in rr_ok and r["Type"] == record_type:
                    return r
            return None
        except (ClientException, ServerException) as e:
            logger.error("获取DNS记录失败: %s", e)
            return None

    def update_domain_record(self, record_id, subdomain, record_type, value, ttl):
        request = UpdateDomainRecordRequest.UpdateDomainRecordRequest()
        request.set_RecordId(record_id)
        request.set_RR(subdomain)
        request.set_Type(record_type)
        request.set_Value(value)
        request.set_TTL(ttl)
        request.set_accept_format("json")
        try:
            self.client.do_action_with_exception(request)
            return True
        except (ClientException, ServerException) as e:
            if "DomainRecordDuplicate" not in str(e):
                logger.error("更新DNS记录失败: %s", e)
            return "DomainRecordDuplicate" in str(e)

    def add_domain_record(self, domain_name, subdomain, record_type, value, ttl):
        request = AddDomainRecordRequest.AddDomainRecordRequest()
        request.set_DomainName(domain_name)
        request.set_RR(subdomain)
        request.set_Type(record_type)
        request.set_Value(value)
        request.set_TTL(ttl)
        request.set_accept_format("json")
        try:
            self.client.do_action_with_exception(request)
            return True
        except (ClientException, ServerException) as e:
            if "DomainRecordDuplicate" not in str(e):
                logger.error("添加DNS记录失败: %s", e)
            return "DomainRecordDuplicate" in str(e)


def get_current_ipv6():
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", "15", CONFIG["IP_SERVICE"]],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0 and out.stdout:
            ip = out.stdout.strip()
            if ":" in ip and len(ip) <= 45:
                return ip
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    try:
        r = requests.get(CONFIG["IP_SERVICE"], timeout=15)
        if r.status_code == 200:
            ip = r.text.strip()
            if ":" in ip and len(ip) <= 45:
                return ip
    except Exception:
        pass
    logger.error("获取 IPv6 失败")
    return None


_last_ip = None


def run_update():
    global _last_ip
    ip = get_current_ipv6()
    if not ip or ip == _last_ip:
        return bool(ip)
    key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID")
    key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
    if not key_id or not key_secret:
        logger.error("未配置 ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET")
        return False
    dns = AliDNSClient(key_id, key_secret, CONFIG["REGION_ID"])
    record = dns.get_domain_records(CONFIG["DOMAIN"], CONFIG["SUBDOMAIN"], "AAAA")
    if record:
        ok = dns.update_domain_record(
            record["RecordId"],
            record.get("RR", CONFIG["SUBDOMAIN"]),
            "AAAA",
            ip,
            CONFIG["TTL"],
        )
    else:
        ok = dns.add_domain_record(
            CONFIG["DOMAIN"], CONFIG["SUBDOMAIN"], "AAAA", ip, CONFIG["TTL"]
        )
    if ok:
        _last_ip = ip
        host = (
            CONFIG["DOMAIN"]
            if CONFIG["SUBDOMAIN"] == "@"
            else f"{CONFIG['SUBDOMAIN']}.{CONFIG['DOMAIN']}"
        )
        logger.info("DNS 更新: %s -> %s", host, ip)
    return ok


def run_loop():
    import fcntl
    import time

    lock_path = os.path.join(
        os.environ.get("TMPDIR", os.environ.get("TEMPDIR", "/tmp")),
        f"ddns-{CONFIG['DOMAIN'].replace('.', '-')}.lock",
    )
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        return
    try:
        while True:
            try:
                run_update()
            except Exception:
                pass
            time.sleep(CONFIG["CHECK_INTERVAL"])
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def main():
    try:
        run_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
