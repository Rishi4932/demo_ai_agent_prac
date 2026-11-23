import json

import re



from dataclasses import dataclass

from typing import Any, Dict, List





@dataclass

class UnderstandingResult:

    success: bool

    task_description: str

    steps: List[Dict[str, Any]]

    message: str





class TaskUnderstanding:

    """

    Converts natural-language user instructions into

    structured executable task steps.



    The component supports:

    1. LLM-based task understanding through a provider.

    2. Deterministic rule-based understanding as a fallback.

    3. Native Windows application tasks.

    4. Basic file-saving tasks.

    5. Explicit new-document creation for supported applications.

    """



    SYSTEM_PROMPT = """

You are the task-understanding component of a Windows

computer-use AI agent.



Your job is to convert a user's natural-language request

into a structured list of executable steps.



Return ONLY valid JSON.



Required format:



{

    "task_description": "short description",

    "steps": [

        {

            "description": "human-readable step",

            "action": "action_type",

            "parameters": {}

        }

    ]

}



Available action types include:



- browser_start

- browser_open_url

- browser_click

- browser_fill

- browser_press

- browser_get_text

- browser_screenshot

- browser_wait

- browser_close

- open_application

- click_control

- type_into_control

- get_control_text

- uia_click_control

- uia_invoke_control

- uia_get_control_text

- click

- double_click

- right_click

- type

- press

- hotkey

- scroll

- wait

- screenshot



Important rules:



1. Preserve the user's requested operations.

2. If the user explicitly asks to create a new blank

   document, include the required keyboard shortcut

   as a separate step.

3. Do not silently remove requested operations.

4. Do not invent unnecessary actions.

5. Do not execute the task.

6. Only understand and structure the task.

"""



    def __init__(self, provider=None):

        self.provider = provider



    def understand(

        self,

        user_instruction: str,

    ) -> UnderstandingResult:



        if not user_instruction.strip():

            return UnderstandingResult(

                success=False,

                task_description="",

                steps=[],

                message="Task instruction is empty.",

            )



        if self.provider is None:

            return self._rule_based_understanding(

                user_instruction

            )



        try:

            response = self.provider.generate(

                system_prompt=self.SYSTEM_PROMPT,

                user_prompt=user_instruction,

            )



            return self._parse_response(

                user_instruction,

                response,

            )



        except Exception as error:

            return UnderstandingResult(

                success=False,

                task_description=user_instruction,

                steps=[],

                message=f"Task understanding failed: {error}",

            )



    # ==========================================================

    # LLM RESPONSE PARSING

    # ==========================================================



    def _parse_response(

        self,

        user_instruction: str,

        response: str,

    ) -> UnderstandingResult:



        try:

            data = json.loads(response)



        except json.JSONDecodeError:

            cleaned = self._extract_json(response)



            try:

                data = json.loads(cleaned)



            except json.JSONDecodeError as error:

                return UnderstandingResult(

                    success=False,

                    task_description=user_instruction,

                    steps=[],

                    message=f"Invalid JSON response: {error}",

                )



        if not isinstance(data, dict):

            return UnderstandingResult(

                success=False,

                task_description=user_instruction,

                steps=[],

                message="LLM response must be a JSON object.",

            )



        steps = data.get("steps", [])



        if not isinstance(steps, list):

            return UnderstandingResult(

                success=False,

                task_description=user_instruction,

                steps=[],

                message="'steps' must be a list.",

            )



        validated_steps = []



        for step in steps:



            if not isinstance(step, dict):

                continue



            description = step.get(

                "description",

                "",

            )



            action = step.get(

                "action",

                "",

            )



            parameters = step.get(

                "parameters",

                {},

            )



            if not description or not action:

                continue



            if not isinstance(parameters, dict):

                parameters = {}



            validated_steps.append({

                "description": description,

                "action": action,

                "parameters": parameters,

            })



        if not validated_steps:

            return UnderstandingResult(

                success=False,

                task_description=user_instruction,

                steps=[],

                message="No valid task steps were produced.",

            )



        return UnderstandingResult(

            success=True,

            task_description=data.get(

                "task_description",

                user_instruction,

            ),

            steps=validated_steps,

            message="Task understood successfully.",

        )



    def _extract_json(

        self,

        response: str,

    ) -> str:



        match = re.search(

            r"\{.*\}",

            response,

            re.DOTALL,

        )



        if match:

            return match.group(0)



        return response



    # ==========================================================

    # RULE-BASED UNDERSTANDING

    # ==========================================================



    def _rule_based_understanding(

        self,

        instruction: str,

    ) -> UnderstandingResult:



        text = instruction.strip()

        lower = text.lower()



        steps: List[Dict[str, Any]] = []



        # ------------------------------------------------------

        # Native Windows application startup

        # ------------------------------------------------------



        application = self._detect_windows_application(

            lower

        )



        if application:

            steps.append({

                "description": (

                    f"Open {application['name']}"

                ),

                "action": "open_application",

                "parameters": {

                    "application": application["command"],

                },

            })



        # ------------------------------------------------------

        # Create a new blank document

        # ------------------------------------------------------

        #

        # This is intentionally added only when the user

        # explicitly asks for a new/blank document.

        #

        # For Notepad and similar applications, Ctrl+N

        # creates a new document.

        #

        # We do NOT add Ctrl+N for every Notepad task,

        # because that would introduce an unnecessary action.

        # ------------------------------------------------------



        new_document_requested = self._contains_any(

            lower,

            [

                "create a new blank document",

                "create a blank document",

                "create a new document",

                "new blank document",

                "start a new blank document",

                "start a new document",

                "open a new blank document",

                "open a new document",

                "make a new blank document",

                "make a new document",

            ],

        )



        if (

            new_document_requested

            and application

        ):

            steps.append({

                "description": (

                    "Create a new blank document"

                ),

                "action": "hotkey",

                "parameters": {

                    "keys": [

                        "ctrl",

                        "n",

                    ],

                },

            })



        # ------------------------------------------------------

        # Browser startup

        # ------------------------------------------------------



        if self._contains_any(

            lower,

            [

                "open chrome",

                "open google chrome",

                "open browser",

                "start chrome",

                "start browser",

                "launch chrome",

                "launch browser",

            ],

        ):

            steps.append({

                "description": "Open the browser",

                "action": "browser_start",

                "parameters": {},

            })



        # ------------------------------------------------------

        # URL detection

        # ------------------------------------------------------



        url_match = re.search(

            r"https?://[^\s]+",

            text,

        )



        if url_match:



            url = url_match.group(0).rstrip(

                ".,)"

            )



            steps.append({

                "description": f"Open {url}",

                "action": "browser_open_url",

                "parameters": {

                    "url": url,

                },

            })



        # ------------------------------------------------------

        # ------------------------------------------------------
        # Website-name navigation
        # ------------------------------------------------------
        # Convert common website names into URLs while preserving the
        # existing explicit-URL handling above.

        website_match = re.search(
            r"\b(?:open|visit|go to|navigate to|browse to)\s+"
            r"(?:the\s+)?(.+?)"
            r"(?:\s+(?:in|using|with)\s+"
            r"(?:google\s+)?chrome)?[.!?]*$",
            text,
            re.IGNORECASE,
        )

        if website_match and not url_match:
            website_name = website_match.group(1).strip()
            website_name = re.sub(
                r"\s+(?:in|using|with)\s+(?:google\s+)?chrome$",
                "",
                website_name,
                flags=re.IGNORECASE,
            ).strip(" .,!? ")

            website_key = re.sub(r"\s+", " ", website_name.lower())
            known_websites = {
                "youtube": "https://www.youtube.com",
                "you tube": "https://www.youtube.com",
                "google": "https://www.google.com",
                "google search": "https://www.google.com",
                "gmail": "https://mail.google.com",
                "google mail": "https://mail.google.com",
                "github": "https://github.com",
                "git hub": "https://github.com",
                "facebook": "https://www.facebook.com",
                "instagram": "https://www.instagram.com",
                "linkedin": "https://www.linkedin.com",
                "reddit": "https://www.reddit.com",
                "wikipedia": "https://www.wikipedia.org",
                "amazon": "https://www.amazon.com",
                "chatgpt": "https://chatgpt.com",
                "openai": "https://openai.com",
                "x": "https://x.com",
                "twitter": "https://x.com",
            }

            website_url = known_websites.get(website_key)

            # Accept explicit domain names even when not listed above.
            if website_url is None and re.fullmatch(
                r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?",
                website_name,
            ):
                website_url = "https://" + website_name

            if website_url:
                steps.append({
                    "description": f"Open website {website_name}",
                    "action": "browser_open_url",
                    "parameters": {"url": website_url},
                })

        # Page reading

        # ------------------------------------------------------



        if self._contains_any(

            lower,

            [

                "read the page",

                "read page",

                "get the page text",

                "read the webpage",

                "read the website",

                "get page text",

                "get webpage text",

            ],

        ):

            steps.append({

                "description": "Read the page text",

                "action": "browser_get_text",

                "parameters": {

                    "selector": "body",

                },

            })



        # ------------------------------------------------------

        # Search / fill

        # ------------------------------------------------------



        search_match = re.search(

            r"(?:search|look up|find)\s+"

            r"(?:for\s+)?[\"']?(.+?)[\"']?"

            r"(?:\s+(?:on|in)\s+"

            r"(?:google|the browser|the website))?$",

            text,

            re.IGNORECASE,

        )



        if search_match:



            search_text = (

                search_match.group(1).strip()

            )



            if search_text:



                steps.append({

                    "description": (

                        f"Search for {search_text}"

                    ),

                    "action": "browser_fill",

                    "parameters": {

                        "selector": (

                            "input[name='q'], "

                            "textarea[name='q'], "

                            "input[type='search']"

                        ),

                        "text": search_text,

                    },

                })



                steps.append({

                    "description": "Submit the search",

                    "action": "browser_press",

                    "parameters": {

                        "selector": (

                            "input[name='q'], "

                            "textarea[name='q'], "

                            "input[type='search']"

                        ),

                        "key": "ENTER",

                    },

                })



        # ------------------------------------------------------

        # Native Windows typing + Save As

        # ------------------------------------------------------



        save_match = re.search(

            r"(?:save|save the file|save it)"

            r"\s+(?:the file\s+)?as\s+"

            r"[\"']?(.+?)[\"']?\s*$",

            text,

            re.IGNORECASE,

        )



        text_without_save = text

        save_path = None



        if save_match and application:



            save_path = save_match.group(1).strip()



            # Remove the save instruction from the

            # portion that will be typed.

            text_without_save = (

                text[:save_match.start()]

                .strip()

            )



            # Remove trailing comma and/or "and".

            text_without_save = re.sub(

                r"\s*,\s*(?:and)?\s*$",

                "",

                text_without_save,

                flags=re.IGNORECASE,

            )



        # ------------------------------------------------------

        # Native Windows typing

        # ------------------------------------------------------



        type_match = re.search(

            r"(?:type|write|enter)\s+(.+)$",

            text_without_save,

            re.IGNORECASE,

        )



        if type_match and application:



            typed_text = (

                type_match.group(1).strip()

            )



            if typed_text:



                # Remove the "create a new document"

                # phrase from the text that gets typed.

                #

                # This protects against instructions such as:

                #

                # "Open Notepad, create a new document,

                #  and type Hello World."

                #

                # Only the actual requested text should

                # reach the type action.

                typed_text = re.sub(

                    r"^(?:"

                    r"create\s+(?:a\s+)?"

                    r"(?:new\s+)?blank\s+document"

                    r"|create\s+(?:a\s+)?new\s+document"

                    r"|start\s+(?:a\s+)?new\s+"

                    r"(?:blank\s+)?document"

                    r"|open\s+(?:a\s+)?new\s+"

                    r"(?:blank\s+)?document"

                    r"),?\s*(?:and\s+)?",

                    "",

                    typed_text,

                    flags=re.IGNORECASE,

                ).strip()



                if typed_text:



                    steps.append({

                        "description": (

                            f"Type '{typed_text}'"

                        ),

                        "action": "type",

                        "parameters": {

                            "text": typed_text,

                        },

                    })



        # ------------------------------------------------------

        # Save As

        # ------------------------------------------------------



        if save_match and application:



            steps.append({

                "description": (

                    "Open Save As dialog"

                ),

                "action": "hotkey",

                "parameters": {

                    "keys": [

                        "ctrl",

                        "shift",

                        "s",

                    ],

                },

            })



            steps.append({

                "description": (

                    f"Enter save path '{save_path}'"

                ),

                "action": "type",

                "parameters": {

                    "text": save_path,

                },

            })



            steps.append({

                "description": (

                    "Confirm file save"

                ),

                "action": "press",

                "parameters": {

                    "key": "ENTER",

                },

            })



        # ------------------------------------------------------

        # Native Windows key press

        # ------------------------------------------------------



        press_match = re.search(

            r"(?:press|hit)\s+"

            r"(enter|escape|esc|tab|space|backspace)"

            r"\s*$",

            lower,

            re.IGNORECASE,

        )



        if press_match and application:



            key = press_match.group(1).upper()



            if key == "ESC":

                key = "ESC"



            steps.append({

                "description": (

                    f"Press {key}"

                ),

                "action": "press",

                "parameters": {

                    "key": key,

                },

            })



        # ------------------------------------------------------

        # Screenshot

        # ------------------------------------------------------



        if self._contains_any(

            lower,

            [

                "screenshot",

                "take a screenshot",

                "capture the screen",

                "capture screenshot",

            ],

        ):

            steps.append({

                "description": (

                    "Take a screenshot"

                ),

                "action": "browser_screenshot",

                "parameters": {

                    "path": (

                        "data/screenshots/"

                        "task_screenshot.png"

                    ),

                },

            })



        # ------------------------------------------------------

        # Browser wait

        # ------------------------------------------------------



        wait_match = re.search(

            r"wait\s+"

            r"(?:for\s+)?"

            r"(\d+(?:\.\d+)?)\s*"

            r"(?:seconds?|sec|s)?",

            lower,

        )



        if wait_match:



            seconds = float(

                wait_match.group(1)

            )



            steps.append({

                "description": (

                    f"Wait for {seconds} seconds"

                ),

                "action": "browser_wait",

                "parameters": {

                    "milliseconds": int(

                        seconds * 1000

                    ),

                },

            })



        # ------------------------------------------------------

        # Close browser

        # ------------------------------------------------------



        if self._contains_any(

            lower,

            [

                "close browser",

                "close chrome",

                "close the browser",

                "close the chrome",

            ],

        ):

            steps.append({

                "description": (

                    "Close the browser"

                ),

                "action": "browser_close",

                "parameters": {},

            })



        # ------------------------------------------------------

        # Result

        # ------------------------------------------------------



        if not steps:

            return UnderstandingResult(

                success=False,

                task_description=text,

                steps=[],

                message=(

                    "The rule-based task-understanding "

                    "fallback could not identify any actions."

                ),

            )



        return UnderstandingResult(

            success=True,

            task_description=text,

            steps=steps,

            message=(

                "Task understood using the "

                "temporary rule-based provider."

            ),

        )



    # ==========================================================

    # WINDOWS APPLICATION DETECTION

    # ==========================================================



    @staticmethod

    def _detect_windows_application(

        text: str,

    ) -> Dict[str, str] | None:



        applications = {



            "notepad": {

                "name": "Notepad",

                "command": "notepad.exe",

            },



            "calculator": {

                "name": "Calculator",

                "command": "calc.exe",

            },



            "calc": {

                "name": "Calculator",

                "command": "calc.exe",

            },



            "paint": {

                "name": "Paint",

                "command": "mspaint.exe",

            },



            "file explorer": {

                "name": "File Explorer",

                "command": "explorer.exe",

            },



            "windows explorer": {

                "name": "File Explorer",

                "command": "explorer.exe",

            },

        }



        for application, details in applications.items():



            if re.search(

                rf"\b{re.escape(application)}\b",

                text,

            ):

                return details



        return None



    # ==========================================================

    # HELPERS

    # ==========================================================



    @staticmethod

    def _contains_any(

        text: str,

        phrases: List[str],

    ) -> bool:



        return any(

            phrase in text

            for phrase in phrases

        )





class MockLLMProvider:

    """

    Test provider that behaves like an LLM.



    This is only for development/testing.

    """



    def __init__(

        self,

        response: str,

    ):

        self.response = response



    def generate(

        self,

        system_prompt: str,

        user_prompt: str,

    ) -> str:



        return self.response