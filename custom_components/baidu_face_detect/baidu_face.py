"""Switch platform that triggers Baidu Face Recognition."""
import asyncio
import json
import logging
import os
from datetime import timedelta

import voluptuous as vol
from homeassistant.components.switch import SwitchEntity, PLATFORM_SCHEMA
from homeassistant.const import CONF_NAME
import homeassistant.helpers.config_validation as cv

_LOGGER = logging.getLogger(__name__)

DOMAIN = "baidu_face_detect"

CONF_PATH = "path"
CONF_APP_ID = "app_id"
CONF_API_KEY = "api_key"
CONF_SECRET_KEY = "secret_key"
CONF_GROUP_ID = "group_id"
CONF_QUALITY = "quality_threshold"

DEFAULT_NAME = "Baidu Face Detect"
DEFAULT_QUALITY = 0.8

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_PATH): cv.isdir,
        vol.Required(CONF_APP_ID): cv.string,
        vol.Required(CONF_API_KEY): cv.string,
        vol.Required(CONF_SECRET_KEY): cv.string,
        vol.Optional(CONF_GROUP_ID, default="default"): cv.string,
        vol.Optional(CONF_QUALITY, default=DEFAULT_QUALITY): vol.Range(0, 1),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    }
)

SCAN_INTERVAL = timedelta(seconds=30)


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the switch."""
    async_add_entities(
        [
            BaiduFaceSwitch(
                name=config[CONF_NAME],
                path=config[CONF_PATH],
                app_id=config[CONF_APP_ID],
                api_key=config[CONF_API_KEY],
                secret_key=config[CONF_SECRET_KEY],
                group_id=config[CONF_GROUP_ID],
                quality_threshold=config[CONF_QUALITY],
            )
        ]
    )


class BaiduFaceSwitch(SwitchEntity):
    """Switch that triggers Baidu Face Recognition."""

    def __init__(
        self,
        name,
        path,
        app_id,
        api_key,
        secret_key,
        group_id,
        quality_threshold,
    ):
        self._name = name
        self._path = path
        self._app_id = app_id
        self._api_key = api_key
        self._secret_key = secret_key
        self._group_id = group_id
        self._quality = quality_threshold
        self._is_on = False
        self._face_result = {}
        # 延迟导入，避免在事件循环外阻塞
        self._client = None

    @property
    def name(self):
        return self._name

    @property
    def is_on(self):
        return self._is_on

    @property
    def extra_state_attributes(self):
        return {"face_result": self._face_result}

    async def async_turn_on(self, **kwargs):
        """Turn on the switch and trigger recognition."""
        self._is_on = True
        await self.async_recognize()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        """Turn off the switch."""
        self._is_on = False
        self._face_result = {}
        self.async_write_ha_state()

    async def async_recognize(self):
        """Run face recognition in executor."""
        if self._client is None:
            from aip import AipFace  # pylint: disable=import-outside-toplevel

            self._client = AipFace(self._app_id, self._api_key, self._secret_key)

        def _run():
            results = {}
            for file_name in os.listdir(self._path):
                full_path = os.path.join(self._path, file_name)
                if not os.path.isfile(full_path):
                    continue
                try:
                    with open(full_path, "rb") as fp:
                        img_data = fp.read()
                    b64 = base64.b64encode(img_data).decode()
                    resp = self._client.multiSearch(
                        b64,
                        "BASE64",
                        self._group_id,
                        {
                            "max_face_num": 10,
                            "match_threshold": 80,
                            "quality_control": "NORMAL",
                            "liveness_control": "LOW",
                        },
                    )
                    if resp.get("error_msg") == "SUCCESS":
                        results[file_name] = resp.get("result", {})
                    else:
                        _LOGGER.warning("Baidu API error: %s", resp)
                except Exception as exc:
                    _LOGGER.error("Error processing %s: %s", full_path, exc)
            return results

        loop = asyncio.get_event_loop()
        import base64  # pylint: disable=import-outside-toplevel

        self._face_result = await loop.run_in_executor(None, _run)
