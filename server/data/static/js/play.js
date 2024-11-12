const bpmInput = document.getElementById("bpm");
const repeatButton = document.getElementById("repeat");

const playButton = document.getElementById("play");
playButton.checked = false;

const songProgress = document.getElementById("progress");
const audio = document.getElementById("audio");

const notes = document.getElementById("notes");
const instruments = ["bass", "snare", "tom", "hihat", "ride", "crash"];

let isPlaying = false;
let repeat = repeatButton.checked;

function playNote(instrument) {
  let audio = new Audio(`/static/audio/${instrument}.mp3`);
  audio.play();
}

function playNoteMatrix() {
  let columns = document.querySelectorAll(".note-column");
  columns.forEach((column, columnIndex) => {
    noteArray = [];
    let delayMs = column.dataset.onset * 1000;

    instruments.forEach((instrument) => {
      if (column.querySelector(`.${instrument} input[type="checkbox"]:checked`) != null) {
        setTimeout(() => {
          playNote(instrument);
        }, delayMs);
      }
    });

    setTimeout(() => {
      column.classList.add("highlighted-column");

      let scrollContainer = notes;
      let containerWidth = scrollContainer.clientWidth;
      let columnLeft = column.offsetLeft;
      let scrollPosition = columnLeft - containerWidth / 4 - 250;

      scrollContainer.scrollTo({
        left: scrollPosition,
        behavior: "smooth",
      });
    }, delayMs);

    setTimeout(() => {
      column.classList.remove("highlighted-column");
    }, delayMs + 250);
  });

  if (repeat) {
    let repeatTimeout = setTimeout(() => {
      playNoteMatrix();
    }, columns[columns.length - 1].dataset.onset * 1000 + 0.25);
  }
}

function updateProgress() {
  if (isPlaying) {
    songProgress.value = (audio.currentTime / audio.duration) * songProgress.max;
    requestAnimationFrame(updateProgress);
  }
}

// listeners
document.addEventListener("keypress", (event) => {
  if (event.key == "r") {
    repeatButton.checked = !repeatButton.checked;
  }
});

repeatButton.addEventListener("change", (event) => {
  repeat = event.target.checked;
});

playButton.addEventListener("click", () => {
  if (isPlaying) {
    isPlaying = false;
    audio.pause();
  } else {
    isPlaying = true;
    playNoteMatrix();
    audio.play();
    updateProgress();
  }
});

repeatButton.addEventListener("change", (event) => {
  repeat = event.target.checked;
});

songProgress.addEventListener("input", () => {
  const newTime = (songProgress.value / songProgress.max) * audio.duration;
  audio.currentTime = newTime;
});

function togglePlayback() {
  if (isPlaying) {
    isPlaying = false;
    playButton.checked = false;
    audio.pause();
  } else {
    playButton.checked = true;
    isPlaying = true;
    audio.play();
    updateProgress();
  }
}

audio.addEventListener("ended", () => {
  if (repeat) {
    isPlaying = true;
    playButton.checked = true;
    audio.play();
    updateProgress();
  } else {
    isPlaying = false;
    playButton.checked = false;
  }
});
