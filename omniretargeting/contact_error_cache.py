"""Read immutable contact-reference errors for a full trajectory or view."""


def reference_contact_errors(validator, task):
    cached = getattr(validator, "fixed_contact_errors", {}).get(id(task))
    if cached is None:
        return task.errors(validator.reference, validator.frame_start)
    first = validator.frame_start - cached[0]
    return cached[1][first : first + len(validator.reference)]
