from django.shortcuts import render, redirect
from .forms import UploadMP3Form
from .models import Song, Note
from .aimodel import transcribe_drums

# Create your views here.

def upload_mp3(request):
    if request.method == 'POST':
        form = UploadMP3Form(request.POST, request.FILES)
        if form.is_valid():
            mp3_file = form.cleaned_data['mp3_file']

            # Process the MP3 file using your AI model
            # Replace 'your_ai_module.process_mp3_file' with your actual AI model function
            note_data = transcribe_drums(mp3_file)

            # Create a new Song instance for the user
            song = Song(title="Your Song Title", owner=request.user, mp3=mp3_file)
            song.save()

            for index, row in note_data.iterrows():
                onset_time = row['onset_time']
                if row['kick']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=36
                    )

                if row['snare']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=38
                    )

                if row['hihat']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=42
                    )

                if row['tom']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=48
                    )

                if row['crash']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=49
                    )

                if row['ride']:
                    Note.objects.create(
                        song=song,
                        timestamp=onset_time,
                        duration=0.1,
                        velocity=100,
                        pitch=51
                    )

            return redirect('song_detail', song_id=song.id)
    else:
        form = UploadMP3Form()
    
    return render(request, 'upload_mp3.html', {'form': form})