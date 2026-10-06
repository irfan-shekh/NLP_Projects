from django import forms


class CommentForm(forms.Form):
    comment = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 6,
            'placeholder': 'Enter your comment here...',
            'id': 'commentInput'
        }),
        required=True,
        min_length=1,
        max_length=5000,
        error_messages={
            'required': 'Please enter a comment to analyze.',
            'min_length': 'Comment must be at least 1 character long.',
            'max_length': 'Comment cannot exceed 5000 characters.'
        }
    )
