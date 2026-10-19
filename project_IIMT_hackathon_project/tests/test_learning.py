import unittest

from project_IIMT_hackathon.learning import (
    ModelServiceError,
    NotesFileError,
    extract_pdf_text,
    retrieve_evidence,
    validate_quiz_payload,
)


NOTES = (
    "Topic: Photosynthesis\n"
    "Light-dependent reactions occur in the thylakoid membrane, where chlorophyll "
    "absorbs sunlight.\n"
    "The Calvin cycle occurs in the stroma and uses ATP and NADPH to convert "
    "carbon dioxide into glucose.\n"
)


class EvidenceRetrievalTests(unittest.TestCase):
    def test_returns_exact_matching_source_line_and_line_number(self):
        result = retrieve_evidence(
            NOTES, "Where do light-dependent reactions occur?"
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["line"], 2)
        self.assertEqual(
            result[0]["text"],
            "Light-dependent reactions occur in the thylakoid membrane, where "
            "chlorophyll absorbs sunlight.",
        )

    def test_returns_no_evidence_for_unsupported_question(self):
        self.assertEqual(
            retrieve_evidence(NOTES, "What is the capital of France?"),
            [],
        )

    def test_ignores_questions_with_only_stop_words(self):
        self.assertEqual(retrieve_evidence(NOTES, "what is it?"), [])

    def test_respects_passage_limit_and_orders_by_relevance(self):
        result = retrieve_evidence(
            NOTES, "Where do light-dependent reactions occur?", limit=1
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["line"], 2)


class PdfExtractionTests(unittest.TestCase):
    def test_extracts_text_and_keeps_pdf_page_number(self):
        content_stream = b"BT /F1 12 Tf 72 720 Td (A text source sentence.) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            (
                f"<< /Length {len(content_stream)} >>\nstream\n".encode()
                + content_stream
                + b"\nendstream"
            ),
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for object_number, body in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{object_number} 0 obj\n".encode())
            pdf.extend(body)
            pdf.extend(b"\nendobj\n")
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010} 00000 n \n".encode())
        pdf.extend(
            (
                f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode()
        )

        extracted = extract_pdf_text(bytes(pdf))

        self.assertIn("[PDF page 1]", extracted)
        self.assertIn("A text source sentence.", extracted)

    def test_rejects_invalid_pdf_with_actionable_message(self):
        with self.assertRaisesRegex(NotesFileError, "could not be read"):
            extract_pdf_text(b"not a PDF")


class QuizValidationTests(unittest.TestCase):
    def setUp(self):
        self.quiz = {
            "question": "Where do light-dependent reactions occur?",
            "options": [
                "Thylakoid membrane",
                "Stroma",
                "Cell nucleus",
                "Mitochondrial matrix",
            ],
            "answer": "Thylakoid membrane",
            "explanation": "The notes place these reactions in the thylakoid membrane.",
            "source_quote": (
                "Light-dependent reactions occur in the thylakoid membrane, "
                "where chlorophyll absorbs sunlight."
            ),
        }

    def test_accepts_quiz_with_answer_and_quote_in_notes(self):
        validated = validate_quiz_payload(self.quiz, NOTES)

        self.assertEqual(validated["answer"], "Thylakoid membrane")
        self.assertEqual(validated["options"], self.quiz["options"])

    def test_rejects_quote_not_present_in_notes(self):
        self.quiz["source_quote"] = "Plants convert sunlight into electricity."

        with self.assertRaises(ModelServiceError):
            validate_quiz_payload(self.quiz, NOTES)

    def test_rejects_answer_not_supported_by_its_quote(self):
        self.quiz["answer"] = "Stroma"

        with self.assertRaises(ModelServiceError):
            validate_quiz_payload(self.quiz, NOTES)

    def test_rejects_duplicate_options(self):
        self.quiz["options"][1] = "Thylakoid membrane"

        with self.assertRaises(ModelServiceError):
            validate_quiz_payload(self.quiz, NOTES)


if __name__ == "__main__":
    unittest.main()
