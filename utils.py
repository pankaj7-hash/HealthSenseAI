from enum import Enum


class LanguageCode(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"


def language_label(language: LanguageCode) -> str:
    labels = {
        LanguageCode.ENGLISH: "English",
        LanguageCode.HINDI: "Hindi",
        LanguageCode.MARATHI: "Marathi",
    }
    return labels.get(language, "English")


def build_system_prompt(language: LanguageCode = LanguageCode.ENGLISH):
    if language == LanguageCode.HINDI:
        return (
            "आप एक स्वास्थ्य सहायक हैं। केवल दिए गए संदर्भ के आधार पर उत्तर दें। "
            "यदि जानकारी उपलब्ध नहीं है, तो कहें कि जानकारी उपलब्ध नहीं है।"
        )

    elif language == LanguageCode.MARATHI:
        return (
            "तुम्ही एक आरोग्य सहाय्यक आहात. फक्त दिलेल्या संदर्भावर आधारित उत्तर द्या. "
            "माहिती उपलब्ध नसेल तर तसे सांगा."
        )

    return (
        "You are a healthcare assistant. Answer only based on the provided context. "
        "If information is not available, say you don't know."
    )